"""
Statistical Tests for TreeRAG Benchmarking.

Implements statistical significance testing:
- Paired t-test
- Wilcoxon signed-rank test
- Bootstrap confidence intervals
- Cohen's d effect size
- Bonferroni correction for multiple comparisons
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum


class TestType(str, Enum):
    """Types of statistical tests."""
    __test__ = False
    PAIRED_TTEST = "paired_ttest"
    WILCOXON = "wilcoxon_signed_rank"
    BOOTSTRAP = "bootstrap"
    PERMUTATION = "permutation"


@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""
    test_type: TestType
    statistic: float
    p_value: float
    significant: bool
    alpha: float = 0.05
    
    # Effect size
    effect_size: Optional[float] = None
    effect_size_interpretation: Optional[str] = None
    
    # Confidence interval
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    ci_level: float = 0.95
    
    # Additional info
    n_samples: int = 0
    method_a_mean: float = 0.0
    method_b_mean: float = 0.0
    mean_difference: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_type": self.test_type.value,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "significant": self.significant,
            "alpha": self.alpha,
            "effect_size": self.effect_size,
            "effect_interpretation": self.effect_size_interpretation,
            "confidence_interval": {
                "lower": self.ci_lower,
                "upper": self.ci_upper,
                "level": self.ci_level
            },
            "n_samples": self.n_samples,
            "method_a_mean": self.method_a_mean,
            "method_b_mean": self.method_b_mean,
            "mean_difference": self.mean_difference
        }
    
    def __str__(self) -> str:
        sig_str = "✓" if self.significant else "✗"
        return (
            f"{self.test_type.value}: stat={self.statistic:.4f}, "
            f"p={self.p_value:.4f} {sig_str}, "
            f"d={self.effect_size:.3f} ({self.effect_size_interpretation})"
        )


@dataclass
class ComparisonSummary:
    """Summary of comparing two methods."""
    method_a: str
    method_b: str
    metric: str
    n_samples: int
    
    # Descriptive statistics
    method_a_mean: float
    method_a_std: float
    method_b_mean: float
    method_b_std: float
    
    # Test results
    tests: Dict[str, StatisticalTestResult] = field(default_factory=dict)
    
    # Overall conclusion
    winner: Optional[str] = None
    confidence: str = "none"  # none, low, medium, high
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "methods": [self.method_a, self.method_b],
            "metric": self.metric,
            "n_samples": self.n_samples,
            self.method_a: {
                "mean": self.method_a_mean,
                "std": self.method_a_std
            },
            self.method_b: {
                "mean": self.method_b_mean,
                "std": self.method_b_std
            },
            "tests": {name: test.to_dict() for name, test in self.tests.items()},
            "winner": self.winner,
            "confidence": self.confidence
        }


class StatisticalTests:
    """
    Statistical testing for benchmark comparisons.
    
    Provides standard tests for comparing retrieval systems:
    - Paired t-test for normally distributed metrics
    - Wilcoxon signed-rank for non-parametric comparison
    - Bootstrap for confidence intervals
    - Effect size calculations
    """
    
    def __init__(self, alpha: float = 0.05, random_seed: int = 42):
        """
        Initialize statistical tests.
        
        Args:
            alpha: Significance level (default: 0.05)
            random_seed: Random seed for bootstrap/permutation tests
        """
        self.alpha = alpha
        self.random_seed = random_seed
        random.seed(random_seed)
    
    def mean(self, values: List[float]) -> float:
        """Compute mean."""
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    def std(self, values: List[float], ddof: int = 1) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        
        m = self.mean(values)
        variance = sum((x - m) ** 2 for x in values) / (len(values) - ddof)
        return math.sqrt(variance)
    
    def paired_ttest(
        self,
        scores_a: List[float],
        scores_b: List[float]
    ) -> StatisticalTestResult:
        """
        Perform paired t-test.
        
        Tests H0: mean(A) = mean(B) vs H1: mean(A) ≠ mean(B)
        
        Args:
            scores_a: Scores for method A
            scores_b: Scores for method B
            
        Returns:
            Statistical test result
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Score lists must have same length")
        
        n = len(scores_a)
        if n < 2:
            return StatisticalTestResult(
                test_type=TestType.PAIRED_TTEST,
                statistic=0.0,
                p_value=1.0,
                significant=False,
                n_samples=n
            )
        
        # Compute differences
        differences = [a - b for a, b in zip(scores_a, scores_b)]
        
        mean_diff = self.mean(differences)
        std_diff = self.std(differences)
        
        # t-statistic
        if std_diff == 0:
            t_stat = float('inf') if mean_diff != 0 else 0.0
        else:
            t_stat = mean_diff / (std_diff / math.sqrt(n))
        
        # Approximate p-value using normal approximation for large n
        # For small n, this is approximate
        p_value = self._ttest_pvalue(abs(t_stat), n - 1)
        
        # Cohen's d effect size
        pooled_std = math.sqrt((self.std(scores_a) ** 2 + self.std(scores_b) ** 2) / 2)
        effect_size = mean_diff / pooled_std if pooled_std > 0 else 0.0
        
        return StatisticalTestResult(
            test_type=TestType.PAIRED_TTEST,
            statistic=t_stat,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=abs(effect_size),
            effect_size_interpretation=self._interpret_cohens_d(abs(effect_size)),
            n_samples=n,
            method_a_mean=self.mean(scores_a),
            method_b_mean=self.mean(scores_b),
            mean_difference=mean_diff
        )
    
    def _ttest_pvalue(self, t_stat: float, df: int) -> float:
        """
        Approximate p-value for t-test using normal approximation.
        
        For more accurate results, use scipy.stats.t.sf
        """
        # Normal approximation for large df
        if df > 30:
            # Use normal CDF approximation
            z = abs(t_stat)
            # Approximation: 2 * (1 - Φ(|t|))
            p = 2 * (1 - self._normal_cdf(z))
            return max(p, 1e-10)
        
        # For small df, use rough approximation
        # This is not as accurate as scipy.stats.t
        t_critical_05 = {
            1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57,
            10: 2.23, 15: 2.13, 20: 2.09, 25: 2.06, 30: 2.04
        }
        
        # Find closest df
        closest_df = min(t_critical_05.keys(), key=lambda k: abs(k - df))
        t_crit = t_critical_05[closest_df]
        
        if abs(t_stat) > t_crit:
            # Rough estimate: more extreme = smaller p
            ratio = t_crit / abs(t_stat)
            return 0.05 * ratio ** 2
        else:
            return 0.05 + (1 - abs(t_stat) / t_crit) * 0.45
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF using error function approximation."""
        # Abramowitz and Stegun approximation
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p = 0.3275911
        
        sign = 1 if x >= 0 else -1
        x = abs(x) / math.sqrt(2)
        
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
        
        return 0.5 * (1.0 + sign * y)
    
    def wilcoxon_signed_rank(
        self,
        scores_a: List[float],
        scores_b: List[float]
    ) -> StatisticalTestResult:
        """
        Perform Wilcoxon signed-rank test (non-parametric).
        
        Tests H0: median(A) = median(B)
        
        Args:
            scores_a: Scores for method A
            scores_b: Scores for method B
            
        Returns:
            Statistical test result
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Score lists must have same length")
        
        n = len(scores_a)
        
        # Compute differences and ranks
        differences = [(a - b, i) for i, (a, b) in enumerate(zip(scores_a, scores_b))]
        
        # Remove zero differences
        non_zero = [(d, i) for d, i in differences if d != 0]
        
        if not non_zero:
            return StatisticalTestResult(
                test_type=TestType.WILCOXON,
                statistic=0.0,
                p_value=1.0,
                significant=False,
                n_samples=n
            )
        
        # Rank by absolute value
        ranked = sorted(non_zero, key=lambda x: abs(x[0]))
        ranks = {}
        for rank, (diff, idx) in enumerate(ranked, 1):
            ranks[idx] = rank
        
        # Compute W+ (sum of positive ranks) and W- (sum of negative ranks)
        w_plus = sum(ranks[i] for d, i in non_zero if d > 0)
        w_minus = sum(ranks[i] for d, i in non_zero if d < 0)
        
        w = min(w_plus, w_minus)
        n_eff = len(non_zero)
        
        # Normal approximation for p-value
        expected = n_eff * (n_eff + 1) / 4
        std_w = math.sqrt(n_eff * (n_eff + 1) * (2 * n_eff + 1) / 24)
        
        if std_w == 0:
            z = 0.0
        else:
            z = (w - expected) / std_w
        
        p_value = 2 * (1 - self._normal_cdf(abs(z)))
        
        # Effect size: r = Z / sqrt(N)
        effect_size = abs(z) / math.sqrt(n) if n > 0 else 0.0
        
        return StatisticalTestResult(
            test_type=TestType.WILCOXON,
            statistic=w,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=effect_size,
            effect_size_interpretation=self._interpret_r(effect_size),
            n_samples=n,
            method_a_mean=self.mean(scores_a),
            method_b_mean=self.mean(scores_b),
            mean_difference=self.mean(scores_a) - self.mean(scores_b)
        )
    
    def bootstrap_ci(
        self,
        scores_a: List[float],
        scores_b: List[float],
        n_bootstrap: int = 10000,
        ci_level: float = 0.95
    ) -> StatisticalTestResult:
        """
        Compute bootstrap confidence interval for mean difference.
        
        Args:
            scores_a: Scores for method A
            scores_b: Scores for method B
            n_bootstrap: Number of bootstrap samples
            ci_level: Confidence level (default: 0.95)
            
        Returns:
            Statistical test result with confidence interval
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Score lists must have same length")
        
        n = len(scores_a)
        observed_diff = self.mean(scores_a) - self.mean(scores_b)
        
        # Bootstrap resampling
        random.seed(self.random_seed)
        bootstrap_diffs = []
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = [random.randint(0, n - 1) for _ in range(n)]
            sample_a = [scores_a[i] for i in indices]
            sample_b = [scores_b[i] for i in indices]
            
            boot_diff = self.mean(sample_a) - self.mean(sample_b)
            bootstrap_diffs.append(boot_diff)
        
        # Compute confidence interval (percentile method)
        bootstrap_diffs.sort()
        alpha_half = (1 - ci_level) / 2
        lower_idx = int(alpha_half * n_bootstrap)
        upper_idx = int((1 - alpha_half) * n_bootstrap)
        
        ci_lower = bootstrap_diffs[lower_idx]
        ci_upper = bootstrap_diffs[upper_idx]
        
        # Check if CI excludes zero
        significant = ci_lower > 0 or ci_upper < 0
        
        # Approximate p-value from bootstrap
        # Count how many bootstrap samples cross zero
        if observed_diff > 0:
            p_value = 2 * sum(1 for d in bootstrap_diffs if d <= 0) / n_bootstrap
        else:
            p_value = 2 * sum(1 for d in bootstrap_diffs if d >= 0) / n_bootstrap
        
        p_value = max(p_value, 1 / n_bootstrap)  # Minimum p-value
        
        # Effect size
        pooled_std = math.sqrt((self.std(scores_a) ** 2 + self.std(scores_b) ** 2) / 2)
        effect_size = abs(observed_diff) / pooled_std if pooled_std > 0 else 0.0
        
        return StatisticalTestResult(
            test_type=TestType.BOOTSTRAP,
            statistic=observed_diff,
            p_value=p_value,
            significant=significant,
            alpha=1 - ci_level,
            effect_size=effect_size,
            effect_size_interpretation=self._interpret_cohens_d(effect_size),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=ci_level,
            n_samples=n,
            method_a_mean=self.mean(scores_a),
            method_b_mean=self.mean(scores_b),
            mean_difference=observed_diff
        )
    
    def permutation_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
        n_permutations: int = 10000
    ) -> StatisticalTestResult:
        """
        Perform permutation test for mean difference.
        
        Args:
            scores_a: Scores for method A
            scores_b: Scores for method B
            n_permutations: Number of permutations
            
        Returns:
            Statistical test result
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Score lists must have same length")
        
        n = len(scores_a)
        observed_diff = abs(self.mean(scores_a) - self.mean(scores_b))
        
        # Combine scores
        combined = list(zip(scores_a, scores_b))
        
        # Count more extreme differences
        random.seed(self.random_seed)
        more_extreme = 0
        
        for _ in range(n_permutations):
            # Randomly swap each pair
            perm_a = []
            perm_b = []
            for a, b in combined:
                if random.random() < 0.5:
                    perm_a.append(a)
                    perm_b.append(b)
                else:
                    perm_a.append(b)
                    perm_b.append(a)
            
            perm_diff = abs(self.mean(perm_a) - self.mean(perm_b))
            if perm_diff >= observed_diff:
                more_extreme += 1
        
        p_value = more_extreme / n_permutations
        p_value = max(p_value, 1 / n_permutations)  # Minimum p-value
        
        # Effect size
        pooled_std = math.sqrt((self.std(scores_a) ** 2 + self.std(scores_b) ** 2) / 2)
        effect_size = observed_diff / pooled_std if pooled_std > 0 else 0.0
        
        return StatisticalTestResult(
            test_type=TestType.PERMUTATION,
            statistic=observed_diff,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=effect_size,
            effect_size_interpretation=self._interpret_cohens_d(effect_size),
            n_samples=n,
            method_a_mean=self.mean(scores_a),
            method_b_mean=self.mean(scores_b),
            mean_difference=self.mean(scores_a) - self.mean(scores_b)
        )
    
    def cohens_d(
        self,
        scores_a: List[float],
        scores_b: List[float]
    ) -> float:
        """
        Compute Cohen's d effect size.
        
        d = (mean_a - mean_b) / pooled_std
        
        Args:
            scores_a: Scores for method A
            scores_b: Scores for method B
            
        Returns:
            Cohen's d effect size
        """
        mean_diff = self.mean(scores_a) - self.mean(scores_b)
        pooled_std = math.sqrt((self.std(scores_a) ** 2 + self.std(scores_b) ** 2) / 2)
        
        if pooled_std == 0:
            return 0.0
        
        return mean_diff / pooled_std
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_r(self, r: float) -> str:
        """Interpret r effect size (for Wilcoxon)."""
        r = abs(r)
        if r < 0.1:
            return "negligible"
        elif r < 0.3:
            return "small"
        elif r < 0.5:
            return "medium"
        else:
            return "large"
    
    def compare_methods(
        self,
        method_a_name: str,
        method_b_name: str,
        scores_a: List[float],
        scores_b: List[float],
        metric_name: str
    ) -> ComparisonSummary:
        """
        Comprehensive comparison of two methods.
        
        Runs multiple statistical tests and provides summary.
        
        Args:
            method_a_name: Name of method A
            method_b_name: Name of method B
            scores_a: Scores for method A
            scores_b: Scores for method B
            metric_name: Name of the metric being compared
            
        Returns:
            Comparison summary with all test results
        """
        summary = ComparisonSummary(
            method_a=method_a_name,
            method_b=method_b_name,
            metric=metric_name,
            n_samples=len(scores_a),
            method_a_mean=self.mean(scores_a),
            method_a_std=self.std(scores_a),
            method_b_mean=self.mean(scores_b),
            method_b_std=self.std(scores_b)
        )
        
        # Run tests
        summary.tests["paired_ttest"] = self.paired_ttest(scores_a, scores_b)
        summary.tests["wilcoxon"] = self.wilcoxon_signed_rank(scores_a, scores_b)
        summary.tests["bootstrap"] = self.bootstrap_ci(scores_a, scores_b)
        summary.tests["permutation"] = self.permutation_test(scores_a, scores_b)
        
        # Determine winner and confidence
        n_significant = sum(1 for t in summary.tests.values() if t.significant)
        
        if n_significant >= 3:
            summary.confidence = "high"
        elif n_significant >= 2:
            summary.confidence = "medium"
        elif n_significant >= 1:
            summary.confidence = "low"
        else:
            summary.confidence = "none"
        
        if summary.confidence != "none":
            if summary.method_a_mean > summary.method_b_mean:
                summary.winner = method_a_name
            else:
                summary.winner = method_b_name
        
        return summary
    
    def bonferroni_correction(
        self,
        p_values: List[float],
        alpha: float = 0.05
    ) -> Tuple[List[bool], float]:
        """
        Apply Bonferroni correction for multiple comparisons.
        
        Args:
            p_values: List of p-values
            alpha: Significance level
            
        Returns:
            Tuple of (significant after correction, corrected alpha)
        """
        n_tests = len(p_values)
        corrected_alpha = alpha / n_tests
        
        significant = [p < corrected_alpha for p in p_values]
        
        return significant, corrected_alpha
    
    def benjamini_hochberg(
        self,
        p_values: List[float],
        alpha: float = 0.05
    ) -> Tuple[List[bool], List[float]]:
        """
        Apply Benjamini-Hochberg FDR correction.
        
        Args:
            p_values: List of p-values
            alpha: Significance level
            
        Returns:
            Tuple of (significant after correction, adjusted p-values)
        """
        n = len(p_values)
        
        # Sort p-values with original indices
        sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])
        
        # Compute adjusted p-values
        adjusted = [0.0] * n
        prev_adj = 0.0
        
        for rank, (idx, p) in enumerate(sorted_pairs, 1):
            adj_p = min(p * n / rank, 1.0)
            adj_p = max(adj_p, prev_adj)  # Enforce monotonicity
            adjusted[idx] = adj_p
            prev_adj = adj_p
        
        significant = [p < alpha for p in adjusted]
        
        return significant, adjusted


def generate_latex_table(comparisons: List[ComparisonSummary]) -> str:
    """
    Generate LaTeX table from comparison summaries.
    
    Args:
        comparisons: List of comparison summaries
        
    Returns:
        LaTeX table code
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Statistical Comparison of Methods}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Metric & Method A & Method B & $p$-value & Cohen's $d$ & Sig. \\",
        r"\midrule"
    ]
    
    for comp in comparisons:
        ttest = comp.tests.get("paired_ttest")
        sig = r"$\checkmark$" if ttest and ttest.significant else r"$\times$"
        p_str = f"{ttest.p_value:.4f}" if ttest else "N/A"
        d_str = f"{ttest.effect_size:.3f}" if ttest else "N/A"
        
        lines.append(
            f"{comp.metric} & {comp.method_a_mean:.3f}$\\pm${comp.method_a_std:.3f} & "
            f"{comp.method_b_mean:.3f}$\\pm${comp.method_b_std:.3f} & "
            f"{p_str} & {d_str} & {sig} \\\\"
        )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    return "\n".join(lines)
