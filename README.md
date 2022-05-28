# Machine learning-based probabilistic ensemble for urban water demand forecasting

## Experiment

> Each step is based on the output from the previous step. So run codes in order. 

- 1. Simulate data: `Run` Data/Sim_AR1. 
- 2. Build component learners:  `Run` Long_term_* and Short_term_*
- 3. Ensemble:
    - `Run` CombineMF.R
    - `Run` Ensemble_short.R and Ensemble_long.R 
    - `Run` aggregated_plot.R 


## Files

- [Data] data used for experiment
- [Ensemble] scripts that conduct ensemble methods
- [LSTM_log] LSTM models
- [MLP_log] MLP models
- [models] cML models
- [Out_dev_MF] validation outputs for ensemble
- [Out_test_MF] test outputs for ensemble
- [Results] report component learner performances
- [Long_term_\*] and [Short_term_\*] component learners
- [Short&Long_term_AR1.R] baseline AR1 model 