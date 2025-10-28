import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

# Consistent plot style
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

# Load and clean data
def load_and_clean_data(filename):
    df = pd.read_csv(filename)
    df['memory_mb'] = df['ru_maxrss'] / 1024 # Convert KB to MB for better readability
    df['cpu_time'] = df['ru_utime'] + df['ru_stime']
    df['execution_time'] = df['rtime']
    df = df[df['status'] == 0] # Only successful runs
    return df

# Calculate summary statistics
def calculate_summary_statistics(df):
    summary = df.groupby(['language', 'task', 'input_size']).agg({
        'execution_time': ['mean', 'std', 'min', 'max'],
        'memory_mb': ['mean', 'std', 'min', 'max'],
        'cpu_time': ['mean', 'std']
    }).round(4)
    return summary

# Calculate coeficient of variation for performance consistency
def calculate_coeficient_of_variation(df):
    cv_data = []
    for (language, task, input_size), group in df.groupby(['language', 'task', 'input_size']):
        time_cv = (group['execution_time'].std() / group['execution_time'].mean()) * 100
        memory_cv = (group['memory_mb'].std() / group['memory_mb'].mean()) * 100
        cv_data.append({
            'language': language,
            'task': task,
            'input_size': input_size,
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
    sns.boxplot(data=df, x='task', y='execution_time', hue='language')
    plt.title('Execution Time Distribution by Data Processing Task and Programming Language')
    plt.xticks(rotation=45)
    plt.ylabel('Execution Time (seconds)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_execution_time_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Memory Usage Boxplot
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='task', y='memory_mb', hue='language')
    plt.title('Memory Usage Distribution by Data Processing Task and Programming Language')
    plt.xticks(rotation=45)
    plt.ylabel('Memory Usage (megabytes(MB))')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_memory_usage_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Performance Consistency Boxplot
    cv_df = calculate_coeficient_of_variation(df)
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=cv_df, x='task', y='time_cv', hue='language')
    plt.title('Performance Consistency (Coefficient of Variation) by Data Processing Task and Programming Language')
    plt.ylabel('Coefficient of Variation (%)')
    plt.xlabel('Programming Language')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_performance_consistency_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate report
def generate_report(df, os_name, output_prefix):
    print(f"\n{'=' * 60}")
    print(f"Data Analysis Report for {os_name}")
    print(f"{'=' * 60}")

    # Summary Statistics
    print("\n1. Summary Statistics:")
    print('-' * 40)
    summary_stats = calculate_summary_statistics(df)
    print(summary_stats)

    # Performance Consistency
    print("\n2. Performance Consistency (Coefficient of Variation):")
    print('-' * 60)
    cv_df = calculate_coeficient_of_variation(df)
    print("Lower CV Percentage = More Consistent Performance")
    print(cv_df.round(2))

    # ANOVA Test Results
    print("\n3. Statistical Significance (ANOVA Test):")
    print('-' * 40)
    anova_dv = perform_anova_test(df)
    if not anova_dv.empty:
        print("A p-value < 0.05 indicates that the programming language choice sifnificantly affects performance.")
        print(anova_dv.round(4))
    else:
        print("Not enough language variety to perform ANOVA tests.")
    
    # Key Findings
    print("\n4. Key Findings:")
    print('-' * 40)
    fastest_langs_by_task = df.groupby(['task']).apply(lambda x: x.loc[x['execution_time'].idxmin(), 'language'])
    print("Fastest Language by Task:")
    for task, lang in fastest_langs_by_task.items():
        print(f" - {task}: {lang}")
    most_consistent_langs_by_task = cv_df.groupby('language')['time_cv'].mean().sort_values()
    print(f"\nMost consistent language (lowest avg CV): {most_consistent_langs_by_task.index[0]} ({most_consistent_langs_by_task.iloc[0]:.1f}%)")
    print(f"Least consistent language (highest avg CV): {most_consistent_langs_by_task.index[-1]} ({most_consistent_langs_by_task.iloc[-1]:.1f}%)")

# Main function
def main(csv_file, os_name, output_prefix):
    df = load_and_clean_data(csv_file)
    create_boxplots(df, output_prefix)
    generate_report(df, os_name, output_prefix)
    df.to_csv(f'{output_prefix}_processed_data.csv', index=False)
    return df

if __name__ == "__main__":
    main('data/Ubuntu Server 24.04.3/c.csv', 'Ubuntu Server 24.04.3', 'ubuntu_c_analysis')