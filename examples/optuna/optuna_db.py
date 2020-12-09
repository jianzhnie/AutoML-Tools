import optuna

## New Study
study_name = 'example-study'  # Unique identifier of the study.
study = optuna.create_study(study_name=study_name, storage='sqlite:///results/example.db')


def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    return (x - 2) ** 2


study.optimize(objective, n_trials=3)

## Resume Study
study = optuna.create_study(study_name='example-study', storage='sqlite:///results/example.db', load_if_exists=True)
study.optimize(objective, n_trials=3)

## Experimental History
df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
print(df)

