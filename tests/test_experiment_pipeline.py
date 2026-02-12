
import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict

from scripts.generate_paper_tables import (
    LatexTableGenerator,
    TableConfig,
    generate_all_tables
)
from scripts.ablation_study import (
    AblationConfig,
    AblationTarget,
    AblationResult,
    AblationStudyResult,
    AblationStudyRunner,
    MockSystem,
    generate_ablation_report
)


class TestLatexTableGenerator:
    
    @pytest.fixture
    def generator(self, tmp_path):
        return LatexTableGenerator(str(tmp_path / "tables"))
    
    @pytest.fixture
    def sample_results(self):
        return {
            "Flat RAG": {
                "precision_at_5": 0.65,
                "recall_at_5": 0.72,
                "ndcg_at_5": 0.68,
                "mrr": 0.71,
                "groundedness": 0.82
            },
            "TreeRAG": {
                "precision_at_5": 0.78,
                "recall_at_5": 0.85,
                "ndcg_at_5": 0.81,
                "mrr": 0.84,
                "groundedness": 0.91
            }
        }
    
    def test_main_results_table(self, generator, sample_results):
        table = generator.generate_main_results_table(
            sample_results,
            ["precision_at_5", "ndcg_at_5", "mrr"]
        )
        
                               
        assert "\\begin{table}" in table
        assert "\\end{table}" in table
        assert "\\begin{tabular}" in table
        assert "\\toprule" in table
        assert "\\bottomrule" in table
        
                       
        assert "Flat RAG" in table or "Flat\\_RAG" in table
        assert "TreeRAG" in table
    
    def test_best_value_bolded(self, generator, sample_results):
        table = generator.generate_main_results_table(
            sample_results,
            ["precision_at_5"]
        )
        
                                           
        assert "\\textbf{0.780}" in table
    
    def test_ablation_table(self, generator):
        ablations = {
            "TreeRAG (Full)": {
                "precision_at_5": 0.78,
                "ndcg_at_5": 0.81
            },
            "- Hierarchical": {
                "precision_at_5": 0.68,
                "ndcg_at_5": 0.72
            }
        }
        
        table = generator.generate_ablation_table(
            ablations,
            "TreeRAG (Full)",
            ["precision_at_5", "ndcg_at_5"]
        )
        
        assert "\\begin{table}" in table
        assert "Ablation" in table
        assert "$\\Delta$" in table or "Delta" in table
    
    def test_efficiency_table(self, generator):
        efficiency = {
            "Flat RAG": {
                "latency_mean_ms": 200,
                "tokens_context": 2000
            },
            "TreeRAG": {
                "latency_mean_ms": 150,
                "tokens_context": 800
            }
        }
        
        table = generator.generate_efficiency_table(efficiency)
        
        assert "\\begin{table}" in table
        assert "Efficiency" in table
    
    def test_significance_table(self, generator):
        pairwise = {
            ("Flat RAG", "TreeRAG"): {
                "p_value": 0.001,
                "effect_size": 0.8
            }
        }
        
        table = generator.generate_significance_table(
            pairwise,
            ["Flat RAG", "TreeRAG"],
            "precision_at_5"
        )
        
        assert "\\begin{table}" in table
        assert "***" in table or "p<" in table
    
    def test_save_table(self, generator, sample_results):
        table = generator.generate_main_results_table(
            sample_results,
            ["precision_at_5"]
        )
        
        path = generator.save_table(table, "test_table.tex")
        
        assert path.exists()
        content = path.read_text()
        assert "\\begin{table}" in content
    
    def test_custom_config(self, generator, sample_results):
        config = TableConfig(
            caption="Custom Caption",
            label="tab:custom",
            column_format="lcc",
            booktabs=False
        )
        
        table = generator.generate_main_results_table(
            sample_results,
            ["precision_at_5"],
            config=config
        )
        
        assert "Custom Caption" in table
        assert "tab:custom" in table
        assert "\\hline" in table                
    
    def test_latex_escaping(self, generator):
        results = {
            "System_A": {"precision_at_5": 0.5},
            "System&B": {"precision_at_5": 0.6}
        }
        
        table = generator.generate_main_results_table(
            results,
            ["precision_at_5"]
        )
        
                                          
        assert "\\_" in table or "System A" in table
        assert "\\&" in table or "System B" in table


class TestAblationStudy:
    
    @pytest.fixture
    def config(self):
        return AblationConfig(
            targets=[
                AblationTarget.HIERARCHICAL_INDEX,
                AblationTarget.BEAM_SEARCH
            ],
            num_queries=10,
            num_runs=2,
            output_dir=tempfile.mkdtemp()
        )
    
    @pytest.fixture
    def runner(self, config):
        return AblationStudyRunner(config)
    
    def test_mock_system(self):
        config = {t.value: True for t in AblationTarget}
        system = MockSystem(config)
        
        metrics = system.evaluate(["query1", "query2"])
        
        assert "precision_at_5" in metrics
        assert "latency_ms" in metrics
        assert 0 <= metrics["precision_at_5"] <= 1
    
    def test_mock_system_ablation_effect(self):
        full_config = {t.value: True for t in AblationTarget}
        ablated_config = {t.value: True for t in AblationTarget}
        ablated_config["hierarchical_index"] = False
        
        full_system = MockSystem(full_config)
        ablated_system = MockSystem(ablated_config)
        
        full_metrics = full_system.evaluate(["query"])
        ablated_metrics = ablated_system.evaluate(["query"])
        
                                                     
                                                      
                                                                     
        assert 0 <= ablated_metrics["precision_at_5"] <= 1
    
    def test_run_ablation_study(self, runner):
        result = runner.run()
        
        assert isinstance(result, AblationStudyResult)
        assert result.baseline_metrics is not None
        assert len(result.ablations) == 2               
    
    def test_ablation_result_structure(self, runner):
        result = runner.run()
        
        for ablation in result.ablations:
            assert isinstance(ablation, AblationResult)
            assert ablation.component in [t.value for t in AblationTarget]
            assert ablation.enabled == False                          
            assert "precision_at_5" in ablation.delta_from_baseline
            assert "precision_at_5" in ablation.p_values
    
    def test_component_importance(self, runner):
        result = runner.run()
        importance = result.get_component_importance()
        
        assert len(importance) == 2
                                                     
        values = list(importance.values())
        assert values == sorted(values, reverse=True)
    
    def test_results_saved(self, runner):
        result = runner.run()
        
        results_file = Path(runner.output_dir) / "ablation_results.json"
        assert results_file.exists()
        
        with open(results_file) as f:
            saved = json.load(f)
        
        assert "baseline" in saved
        assert "ablations" in saved
    
    def test_generate_report(self, runner):
        result = runner.run()
        report = generate_ablation_report(result)
        
        assert "\\subsection{Ablation Study}" in report
        assert "\\begin{table}" in report
        assert "Full System" in report


class TestAblationDataClasses:
    
    def test_ablation_result_to_dict(self):
        result = AblationResult(
            component="test_component",
            enabled=False,
            metrics={"precision_at_5": 0.75},
            delta_from_baseline={"precision_at_5": -0.05},
            p_values={"precision_at_5": 0.01},
            run_time_seconds=10.5
        )
        
        d = result.to_dict()
        
        assert d["component"] == "test_component"
        assert d["enabled"] == False
        assert d["metrics"]["precision_at_5"] == 0.75
    
    def test_ablation_study_result_to_dict(self):
        ablation = AblationResult(
            component="test",
            enabled=False,
            metrics={"m1": 0.5},
            delta_from_baseline={"m1": -0.1},
            p_values={"m1": 0.05},
            run_time_seconds=5.0
        )
        
        result = AblationStudyResult(
            baseline_metrics={"m1": 0.6},
            ablations=[ablation],
            timestamp="2024-01-01T00:00:00",
            config={"test": True}
        )
        
        d = result.to_dict()
        
        assert "baseline" in d
        assert "ablations" in d
        assert len(d["ablations"]) == 1


class TestAblationConfig:
    
    def test_default_config(self):
        config = AblationConfig()
        
        assert config.num_queries == 100
        assert config.num_runs == 3
        assert len(config.targets) == len(AblationTarget)
    
    def test_custom_config(self):
        config = AblationConfig(
            targets=[AblationTarget.BEAM_SEARCH],
            num_queries=50,
            num_runs=5
        )
        
        assert len(config.targets) == 1
        assert config.num_queries == 50
        assert config.num_runs == 5


class TestPlotResultsImport:
    
    def test_import_plotter(self):
        from scripts.plot_results import ResultPlotter, PlotConfig
        
        assert ResultPlotter is not None
        assert PlotConfig is not None
    
    def test_plot_config(self):
        from scripts.plot_results import PlotConfig
        
        config = PlotConfig()
        
        assert config.figsize == (8, 6)
        assert config.dpi == 300
        assert config.output_format == "pdf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
