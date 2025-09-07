import optuna

from RunHP import train_and_evaluate
from hyperparameter import parse_args


def objective(trial):
    args = parse_args()

    args.alpha = trial.suggest_categorical('alpha', [0.0, 0.25, 0.5, 0.75])
    args.beta = trial.suggest_categorical('beta', [0.25, 0.5, 0.75, 1.0])
    args.eta = trial.suggest_categorical('eta', [0.0, 0.25, 0.5, 0.75, 1.0])
    args.hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128])
    args.out_channels = trial.suggest_categorical('out_channels', [32, 64, 128])
    args.lr = trial.suggest_categorical('lr', [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01])
    args.num_layers = trial.suggest_categorical('num_layers', [1, 2, 3])
    args.rnum_layers = trial.suggest_categorical('rnum_layers', [1, 2, 3])
    val_score = train_and_evaluate(args)
    print(val_score)
    
    return val_score


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"{key}: {value}")

