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
    for (language, task, input_size), group in df.groupby(['language', 'task', 'input_size']):
        if len(group['os'].unique()) > 1:
            os_means = group.groupby('os')['execution_time'].mean()   
            if os_means.mean() > 0:
                cross_os_cv = (os_means.std() / os_means.mean()) * 100
            else:
                cross_os_cv = 0
            cross_os_results.append({
                'language': language,
                'task': task,
                'input_size': input_size,
                'task_with_size': f"{task}-{input_size}",
                'cross_os_cv': cross_os_cv,
                'os_count': len(group['os'].unique()),
                'os_list': ', '.join(group['os'].unique()),
                'time_range': os_means.max() - os_means.min(),
                'fastest_os': os_means.idxmin(),
                'slowest_os': os_means.idxmax()
            })
    return pd.DataFrame(cross_os_results)

# Calculate summary statistics
def calculate_summary_statistics(df):
    summary = df.groupby(['language', 'task', 'input_size']).agg({
        'execution_time': ['mean', 'std', 'min', 'max'],
        'memory_mb': ['mean', 'std', 'min', 'max'],
        'cpu_time': ['mean', 'std']
    }).round(4)
    return summary

# Calculate coefficient of variation for performance consistency
def calculate_coefficient_of_variation(df):
    cv_data = []
    for (language, task, input_size), group in df.groupby(['language', 'task', 'input_size']):
        if group['execution_time'].mean() > 0:
            time_cv = (group['execution_time'].std() / group['execution_time'].mean()) * 100
        else:
            time_cv = 0
        if group['memory_mb'].mean() > 0:
            memory_cv = (group['memory_mb'].std() / group['memory_mb'].mean()) * 100
        else:
            memory_cv = 0    
        cv_data.append({
            'language': language,
            'task': task,
            'input_size': input_size,
            'task_with_size': f"{task}-{input_size}",
            'time_cv': time_cv,
            'memory_cv': memory_cv,
            'time_std': group['execution_time'].std(),
            'time_mean': group['execution_time'].mean(),
        })
    return pd.DataFrame(cv_data)
    
# ANOVA test for statistical significance
def perform_anova_test(df):
    anova_results = []
    for task in df['task'].unique():
        task_data = df[df['task'] == task]
        for input_size in task_data['input_size'].unique():
            size_data = task_data[task_data['input_size'] == input_size]
            if len(size_data['language'].unique()) > 1:
                groups = [group['execution_time'].values for name, group in size_data.groupby('language')]
                if len(groups) > 1:
                    f_stat, p_val = stats.f_oneway(*groups)
                    anova_results.append({
                        'task': task,
                        'input_size': input_size,
                        'f_statistic': f_stat,
                        'p_value': p_val,
                        'significant': p_val < 0.05,
                        'languages_compared': len(size_data['language'].unique())
                    })
    return pd.DataFrame(anova_results)

# Create boxplots for visualization
def create_boxplots(df, output_prefix):
    # Execution Time Boxplot
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='task_with_size', y='execution_time', hue='language')
    plt.title('Execution Time Distribution by Data Processing Task and Programming Language')
    plt.xticks(rotation=45)
    plt.ylabel('Execution Time (seconds)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_execution_time_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Memory Usage Boxplot
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='task_with_size', y='memory_mb', hue='language')
    plt.title('Memory Usage Distribution by Data Processing Task and Programming Language')
    plt.xticks(rotation=45)
    plt.ylabel('Memory Usage (megabytes(MB))')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_memory_usage_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Performance Consistency Boxplot
    cv_df = calculate_coefficient_of_variation(df)
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=cv_df, x='task_with_size', y='time_cv', hue='language')
    plt.title('Performance Consistency (Coefficient of Variation) by Data Processing Task and Input Size')
    plt.ylabel('Coefficient of Variation (%)')
    plt.xlabel('Task and Input Size')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_performance_consistency_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

# Cross-OS Visualization
def create_cross_os_plots(df, output_prefix):
    # Execution Time across OSs
    plt.figure(figsize=(20, 12))
    task_sizes = sorted(df['task_with_size'].unique())
    n_plots = len(task_sizes)
    cols = 3
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten() if rows > 1 else [axes]
    
    for i, task_size in enumerate(task_sizes):
        if i < len(axes):
            task_data = df[df['task_with_size'] == task_size]
            sns.boxplot(data=task_data, x='os', y='execution_time', hue='language', ax=axes[i])
            axes[i].set_title(f'{task_size}')
            axes[i].set_ylabel('Execution Time (seconds)')
            axes[i].tick_params(axis='x', rotation=45)
            if i > 0:
                axes[i].get_legend().remove()  
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.suptitle('Execution Time by OS and Language (Separated by Task and Input Size)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_cross_os_execution_time.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Performance Ranking Across OSs
    performance_matrix = df.groupby(['language', 'task_with_size', 'os'])['execution_time'].mean().reset_index()
    os_list = df['os'].unique()
    n_os = len(os_list)
    cols = min(3, n_os)
    rows = (n_os + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, os_name in enumerate(os_list):
        if i < len(axes):
            os_data = performance_matrix[performance_matrix['os'] == os_name]
            heatmap_data = os_data.pivot_table(
                values='execution_time', 
                index='language', 
                columns='task_with_size', 
                aggfunc='mean'
            ).fillna(0)
            
            sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='viridis', ax=axes[i])
            axes[i].set_title(f'Performance on {os_name}\n(Lower = Better)')
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.suptitle('Performance Ranking: Language vs Task-Size vs OS', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_cross_os_performance_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Cross-OS Consistency Heatmap
    cross_os_df = calculate_cross_os_consistency(df)
    if not cross_os_df.empty:
        plt.figure(figsize=(16, 8))
        
        # Create a grouped bar plot
        sns.barplot(data=cross_os_df, x='task_with_size', y='cross_os_cv', hue='language')
        plt.title('Cross-OS Performance Consistency by Language and Task-Size\n(Lower CV = More Consistent Across OSs)')
        plt.ylabel('Coefficient of Variation (%)')
        plt.xlabel('Task and Input Size')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_cross_os_consistency.png', dpi=300, bbox_inches='tight')
        plt.close()

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

    # Cross-OS Consistency Analysis
    report_lines.append("\nCross-OS Performance Consistency:")
    report_lines.append('-' * 60)
    report_lines.append("Lower Cross-OS CV = More Consistent Performance Across Operating Systems")
    cross_os_df = calculate_cross_os_consistency(df)
    if not cross_os_df.empty:
        cross_os_sorted = cross_os_df.sort_values('cross_os_cv').round(2)
        report_lines.append(cross_os_sorted.to_string())
        report_lines.append("\nKey Cross-OS Findings:")
        report_lines.append('-' * 40)
        most_consistent = cross_os_sorted.iloc[0]
        least_consistent = cross_os_sorted.iloc[-1]
        
        report_lines.append(f"Most consistent across OSs: {most_consistent['language']} - {most_consistent['task']} (CV: {most_consistent['cross_os_cv']}%)")
        report_lines.append(f"Least consistent across OSs: {least_consistent['language']} - {least_consistent['task']} (CV: {least_consistent['cross_os_cv']}%)")
        
        report_lines.append("\nOS Performance Rankings:")
        os_performance = df.groupby('os')['execution_time'].mean().sort_values()
        for i, (os_name, avg_time) in enumerate(os_performance.items(), 1):
            report_lines.append(f"{i}. {os_name}: {avg_time:.2f}s average")
            
    else:
        report_lines.append("Not enough cross-OS data for analysis.")

    # Summary Statistics
    report_lines.append("\n1. Summary Statistics:")
    report_lines.append('-' * 40)
    summary_stats = calculate_summary_statistics(df)
    report_lines.append(summary_stats.to_string())

    # Performance Consistency
    report_lines.append("\n2. Performance Consistency (Coefficient of Variation):")
    report_lines.append('-' * 60)
    cv_df = calculate_coefficient_of_variation(df)
    report_lines.append("Lower CV Percentage = More Consistent Performance")
    report_lines.append(cv_df.round(2).to_string())

    # ANOVA Test Results
    report_lines.append("\n3. Statistical Significance (ANOVA Test):")
    report_lines.append('-' * 40)
    anova_df = perform_anova_test(df)
    if not anova_df.empty:
        report_lines.append("A p-value < 0.05 indicates that the programming language choice significantly affects performance.")
        report_lines.append(anova_df.round(4).to_string())
    else:
        report_lines.append("Not enough language variety to perform ANOVA tests.")
    
    # Key Findings
    report_lines.append("\n4. Key Findings:")
    report_lines.append('-' * 40)
    fastest_langs_by_task = df.groupby(['task']).apply(lambda x: x.loc[x['execution_time'].idxmin(), 'language'])
    report_lines.append("Fastest Language by Task:")
    for task, lang in fastest_langs_by_task.items():
        report_lines.append(f" - {task}: {lang}")
    
    most_consistent_langs = cv_df.groupby('language')['time_cv'].mean().sort_values()
    report_lines.append(f"\nMost consistent language (lowest avg CV): {most_consistent_langs.index[0]} ({most_consistent_langs.iloc[0]:.1f}%)")
    report_lines.append(f"Least consistent language (highest avg CV): {most_consistent_langs.index[-1]} ({most_consistent_langs.iloc[-1]:.1f}%)")

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
    create_cross_os_plots(df, output_prefix)
    create_boxplots(df, output_prefix)
    generate_report(df, output_prefix)
    df.to_csv(f'{output_prefix}_processed_combined_data.csv', index=False)
    sys.stdout = saver.terminal
    saver.close()
    return df

if __name__ == "__main__":
    main('data/All_languages_combined_data/combined_python_C_data.csv', 'python3_and_C_cross_os_analysis')