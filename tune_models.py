import mlflow

def tune_models():
    experiments = mlflow.search_runs(
        run_view_type=1)
    experiments = experiments.sort_values(
        by='metrics.f_test', ascending=False).head(3)
    print(experiments[['params.learner', 'metrics.f_test']])

if __name__ == '__main__':
    tune_models()
