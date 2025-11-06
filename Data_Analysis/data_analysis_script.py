import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
from scipy import stats

warnings.filterwarnings('ignore')

# Consistent plot style
plt.style.use('default')
sns.set_palette('husl')
sns.set_style('whitegrid')

# Load and clean data
def load_and_clean_data(filename):
    df = pd.read_csv(filename)
    df['memory_mb'] = df['ru_maxrss'] / 1024 # Convert KB to MB for better readability
    df['cpu_time'] = df['ru_utime'] + df['ru_stime']
    df['execution_time'] = df['rtime']
    df = df[df['status'] == 0] # Only successful runs
    df['task_with_size'] = df['task'] + '-' + df['input_size'].astype(str)
    return df

# Calculate Cross-OS Consistency
def calculate_cross_os_consistency(df):
    cross_os_results = []
    metrics = ['execution_time', 'memory_mb']

    for (language, task, input_size), group in df.groupby(['language', 'task', 'input_size']):
        if len(group['os'].unique()) > 1:
            result = {
                'language': language,
                'task': task,
                'input_size': input_size,
                'task_with_size': f"{task}-{input_size}",
                'os_count': len(group['os'].unique()),
                'os_list': ', '.join(group['os'].unique())
            }
            
            for metric in metrics:
                os_means = group.groupby('os')[metric].mean()  
                if os_means.mean() > 0:
                    cross_os_cv = (os_means.std() / os_means.mean()) * 100
                    range_pct = ((os_means.max() - os_means.min()) / os_means.mean()) * 100
                else:
                    cross_os_cv = 0
                    range_pct = 0

                result[f'{metric}_cv'] = cross_os_cv
                result[f'{metric}_range_pct'] = range_pct
                result[f'{metric}_min_os'] = os_means.idxmin()
                result[f'{metric}_max_os'] = os_means.idxmax()
                result[f'{metric}_min_value'] = os_means.min()
                result[f'{metric}_max_value'] = os_means.max()
                result[f'{metric}_os_means_mean'] = os_means.mean()
            cross_os_results.append(result)

    return pd.DataFrame(cross_os_results)

# Calculate coefficient of variation for performance consistency
def calculate_coefficient_of_variation(df):
    cv_data = []
    metrics = ['execution_time', 'memory_mb']

    for (language, task, input_size, os), group in df.groupby(['language', 'task', 'input_size', 'os']):
        result = {
            'language': language,
            'task': task,
            'input_size': input_size,
            'os': os,
            'task_with_size': f"{task}-{input_size}",
            'sample_size': len(group)
        }
        
        for metric in metrics:
            values = group[metric].dropna()
            if len(values) > 1 and values.mean() > 0:
                cv = (values.std() / values.mean()) * 100
            else:
                cv = 0
            result[f'{metric}_cv'] = cv
            result[f'{metric}_mean'] = values.mean()
            result[f'{metric}_std'] = values.std()
        
        cv_data.append(result)
    
    return pd.DataFrame(cv_data)

# Cross OS ANOVA test
def perform_cross_os__tests(df):
    results = []
    metrics = ['execution_time', 'memory_mb']
    
    for metric in metrics:
        for (language, task, input_size), group in df.groupby(['language', 'task', 'input_size']):
            if len(group['os'].unique()) > 1:
                # Prepare groups for ANOVA
                groups = []
                os_names = []
                for os_name, os_group in group.groupby('os'):
                    valid_values = os_group[metric].dropna()
                    if len(valid_values) > 0:
                        groups.append(valid_values)
                        os_names.append(os_name)
                
                if len(groups) > 1:
                    # ANOVA test
                    f_stat, p_val = stats.f_oneway(*groups)
                    
                    # Kruskal-Wallis test (non-parametric alternative)
                    try:
                        h_stat, kw_p_val = stats.kruskal(*groups)
                    except:
                        h_stat, kw_p_val = np.nan, np.nan
                    
                    result = {
                        'metric': metric,
                        'language': language,
                        'task': task,
                        'input_size': input_size,
                        'os_count': len(groups),
                        'os_list': ', '.join(os_names),
                        'f_statistic': f_stat,
                        'p_value': p_val,
                        'h_statistic': h_stat,
                        'kw_p_value': kw_p_val,
                        'significant_anova': p_val < 0.05,
                        'significant_kw': kw_p_val < 0.05 if not np.isnan(kw_p_val) else False
                    }
                    
                    results.append(result)
    
    return pd.DataFrame(results)

# Graph with OS, Language, and Task
def create_comprehensive_plots(df, output_prefix):
    # Multi-metric facet grid
    metrics = ['execution_time', 'memory_mb']
    metric_names = ['Execution Time (seconds)', 'Memory Usage (MB)']
    for metric, metric_name in zip(metrics, metric_names):
        tasks = df['task'].unique()
        n_tasks = len(tasks)
        cols = min(3, n_tasks)
        rows = (n_tasks + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_tasks == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, task in enumerate(tasks):
            if i < len(axes):
                task_data = df[df['task'] == task]
                sns.boxplot(data=task_data, x='os', y=metric, hue='language', 
                           hue_order=df['language'].unique(), ax=axes[i])
                axes[i].set_title(f'{task}')
                axes[i].set_xlabel('Operating System')
                axes[i].set_ylabel(metric_name)
                axes[i].tick_params(axis='x', rotation=45)
                if metric == 'execution_time':
                    axes[i].set_yscale('log')
                if i > 0:
                    axes[i].get_legend().remove()
                else:
                    axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Language')
        
        for i in range(len(tasks), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'{metric_name.split(" ")[0]} by OS and Language for Each Data Processing Task', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_{metric}_facet_os_language_task.png', dpi=300, bbox_inches='tight')
        plt.close()

    create_cross_os_consistency_plots(df, output_prefix)
    create_os_performance_profile(df, output_prefix)

# Cross-OS Consistency Plots
def create_cross_os_consistency_plots(df, output_prefix):
    cross_os_df = calculate_cross_os_consistency(df)
    if cross_os_df.empty:
        print("No cross-OS data available for consistency plots.")
        return
    metrics = ['execution_time', 'memory_mb']
    metric_names = ['Execution Time', 'Memory Usage']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        cv_col = f'{metric}_cv'
        if cv_col in cross_os_df.columns:
            pivot_cv = cross_os_df.pivot_table(values=cv_col, index='task', columns='language', aggfunc='mean')
            sns.heatmap(pivot_cv, annot=True, fmt=".1f", cmap='viridis_r', ax=axes[i],cbar_kws={'label': 'Coefficient of Variation (%)'})
            axes[i].set_title(f'Cross-OS Consistency of {metric_name}. Lower CV = More Consistent')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_cross_os_consistency_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Cross OS Consistency by Language
    plt.figure(figsize=(12, 8))
    metric_data = []
    for metric, metric_name in zip(metrics, metric_names):
        cv_col = f'{metric}_cv'
        if cv_col in cross_os_df.columns:
            for language in cross_os_df['language'].unique():
                lang_data = cross_os_df[cross_os_df['language'] == language]
                if not lang_data.empty:
                    metric_data.append({
                        'language': language,
                        'cv_mean': lang_data[cv_col].mean(),
                        'cv_std': lang_data[cv_col].std(),
                        'metric': metric_name
                    })
    if metric_data:
        metric_df = pd.DataFrame(metric_data)
        sns.barplot(data=metric_df, x='language', y='cv_mean', hue='metric')
        plt.title('Average Cross-OS Consistency by Language and Metric')
        plt.ylabel('Coefficient of Variation (%)')
        plt.xlabel('Programming Language')
        plt.legend(title='Metric')
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_cross_os_consistency_by_language.png', dpi=300, bbox_inches='tight')
        plt.close()

# OS Performance Profile Plots
def create_os_performance_profile(df, output_prefix):
    metrics = ['execution_time', 'memory_mb']
    metric_names = ['Execution Time', 'Memory Usage']
    performance_data = []
    for metric in metrics:
        for (task, language), group in df.groupby(['task', 'language']):
            if len(group['os'].unique()) > 1:
                min_value = group[metric].min()
                if min_value > 0:
                    for os_name, os_group in group.groupby('os'):
                        normalized_perf = os_group[metric].mean() / min_value
                        performance_data.append({
                            'os': os_name,
                            'task': task,
                            'language': language,
                            'metric': metric.replace('_', ' ').title(),
                            'normalized_performance': normalized_perf,
                            'raw_performance': os_group[metric].mean()
                        })
    if performance_data:
        perf_df = pd.DataFrame(performance_data)

        plt.figure(figsize=(12, 8))
        sns.boxplot(data=perf_df, x='os', y='normalized_performance', hue='metric')
        plt.axhline(1, color='red', linestyle='--', label='Best Performance Baseline')
        plt.title('OS Performance Profile (Normalized, lower is better)')
        plt.ylabel('Normalized Performance')
        plt.xlabel('Operating System')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_os_performance_profile.png', dpi=300, bbox_inches='tight')
        plt.close()

# Memory visualizations
def create_memory_visualization(df, output_prefix):
    # Memory Usage by OS and Task
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df, x='task', y='memory_mb', hue='os')
    plt.title('Memory Usage by Data Processing Task and Operating System')
    plt.xticks(rotation=45)
    plt.ylabel('Memory Usage (megabytes(MB))')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Operating System')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_memory_usage_by_os_task.png', dpi=300, bbox_inches='tight')
    plt.close()

# Run all visualizations
def create_visualizations(df, output_prefix):
    create_comprehensive_plots(df, output_prefix)
    create_memory_visualization(df, output_prefix)

# Helper class to save report to file
class ReportSaver:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

# Generate report
def generate_report(df, output_prefix):
    report_lines = []
    
    report_lines.append(f"\n{'=' * 60}")
    report_lines.append(f"Cross Operating System Data Analysis Report")
    report_lines.append(f"{'=' * 60}")

    # Dataset Overview
    report_lines.append("\nDataset Overview:")
    report_lines.append('-' * 40)
    report_lines.append(f"Total records: {len(df)}")
    report_lines.append(f"Operating Systems: {', '.join(df['os'].unique())}")
    report_lines.append(f"Languages: {', '.join(df['language'].unique())}")
    report_lines.append(f"Tasks: {', '.join(df['task'].unique())}")
    report_lines.append(f"Input sizes: {', '.join(sorted(df['input_size'].astype(str).unique()))}")

    # Cross-OS Consistency
    report_lines.append("\nCross-OS Consistency:")
    report_lines.append('-' * 50)
    cross_os_df = calculate_cross_os_consistency(df)

    if not cross_os_df.empty:
        metrics = ['execution_time', 'memory_mb']
        metric_names = ['Execution Time', 'Memory Usage']
        for metric, metric_name in zip(metrics, metric_names):
            cv_col = f'{metric}_cv'
            report_lines.append(f"\n{metric_name} Cross-OS Consistency:")
            report_lines.append("-" * 50)

            avg_cv = cross_os_df[cv_col].mean()
            report_lines.append(f"Average Cross-OS Coefficient of Variation (CV): {avg_cv:.1f}%")
            lang_consistency = cross_os_df.groupby('language')[cv_col].mean().sort_values()
            report_lines.append(f"\nConsistency by Language (Lower CV = More Consistent):")
            for lang, cv in lang_consistency.items():
                report_lines.append(f" - {lang}: {cv:.1f}%")
            if not cross_os_df[cv_col].isna().all():
                most_consistent = cross_os_df.loc[cross_os_df[cv_col].idxmin()]
                least_consistent = cross_os_df.loc[cross_os_df[cv_col].idxmax()]

                report_lines.append(f"\nMost Consistent {metric_name}:")
                report_lines.append(f"  {most_consistent['task_with_size']} ({most_consistent['language']})")
                report_lines.append(f"  CV: {most_consistent[cv_col]:.1f}% across {most_consistent['os_list']}")
                report_lines.append(f"\nLeast Consistent {metric_name}:")
                report_lines.append(f"  {least_consistent['task_with_size']} ({least_consistent['language']})")
                report_lines.append(f"  CV: {least_consistent[cv_col]:.1f}% across {least_consistent['os_list']}")
    else:
        report_lines.append("No cross-OS data available for consistency analysis.")

    # ANOVA and Kruskal-Wallis Test Results
    report_lines.append("\nStatistical Significance (ANOVA Test):")
    report_lines.append('-' * 50)
    stats_df = perform_cross_os__tests(df)
    if not stats_df.empty:
       for metric in ['execution_time', 'memory_mb']:
            metric_data = stats_df[stats_df['metric'] == metric]
            if not metric_data.empty:
                significant_count = metric_data['significant_anova'].sum()
                total_tests = len(metric_data)
                
                report_lines.append(f"\n{metric.replace('_', ' ').title()}:")
                report_lines.append(f"  Significant OS differences in {significant_count}/{total_tests} "f"cases ({significant_count/total_tests*100:.1f}%)")
    else:
        report_lines.append("No statistical test results available.")
    
    # Key Findings
    report_lines.append("\nKey Findings:")
    report_lines.append('-' * 50)
    
    if not cross_os_df.empty:
        # Overall consistency ranking
        consistency_ranking = {}
        for metric in ['execution_time', 'memory_mb']:
            cv_col = f'{metric}_cv'
            lang_consistency = cross_os_df.groupby('language')[cv_col].mean().sort_values()
            if not lang_consistency.empty:
                consistency_ranking[metric] = lang_consistency
        
        report_lines.append("\nOverall Cross-OS Consistency Ranking:")
        for metric, scores in consistency_ranking.items():
            report_lines.append(f"\n{metric.replace('_', ' ').title()}:")
            for i, (lang, score) in enumerate(scores.items()):
                report_lines.append(f"  {i+1}. {lang}: {score:.1f}% CV")
        
        # OS Performance Characteristics
        report_lines.append("\nOS PERFORMANCE CHARACTERISTICS:")
        os_performance = df.groupby('os').agg({
            'execution_time': 'mean',
            'memory_mb': 'mean'
        }).round(3)
        
        for os_name in os_performance.index:
            time = os_performance.loc[os_name, 'execution_time']
            memory = os_performance.loc[os_name, 'memory_mb']
            report_lines.append(f"  {os_name}: Time={time}s, Memory={memory}MB")

    if not cross_os_df.empty:
        # Find most consistent combinations
        consistent_combinations = cross_os_df.nsmallest(3, 'execution_time_cv')
        report_lines.append("Most Consistent Cross-OS Performance:")
        for _, combo in consistent_combinations.iterrows():
            report_lines.append(f"  - Use {combo['language']} for {combo['task']} tasks")
        
        # Find best performing OS
        fastest_os = df.groupby('os')['execution_time'].mean().idxmin()
        most_efficient_os = df.groupby('os')['memory_mb'].mean().idxmin()
        report_lines.append(f"\nOS Performance:")
        report_lines.append(f"  - Fastest overall OS: {fastest_os}")
        report_lines.append(f"  - Most memory-efficient OS: {most_efficient_os}")

    # Save report to file
    report_filename = f'{output_prefix}_report.txt'
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    # Print to console
    print('\n'.join(report_lines))

    return report_filename

# Main function
def main(csv_file, output_prefix):
    console_log_file = f'{output_prefix}_console_output.txt'
    saver = ReportSaver(console_log_file)
    sys.stdout = saver
    df = load_and_clean_data(csv_file)
    create_visualizations(df, output_prefix)
    generate_report(df, output_prefix)
    df.to_csv(f'{output_prefix}_processed_combined_data.csv', index=False)
    sys.stdout = saver.terminal
    saver.close()
    return df

if __name__ == "__main__":
    main('data/All_languages_combined_data/combined_python3_C_java_data.csv', 'python3_C_java_cross_os_analysis')