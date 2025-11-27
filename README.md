Diabetes Digital Twin - Code Bundle
===================================
Files included:
- utils.py         : helper functions for data loading & preprocessing
- static_model.py  : trains static risk models (sklearn)
- cgm_lstm.py      : LSTM/GRU model stub for CGM forecasting (tensorflow optional)
- rl_agent.py      : simple RL agent skeleton (Q-learning stub)
- ed_risk.py       : ED risk model (ensemble stub)
- requirements.txt : suggested packages
- example_usage.ipynb : (optional) suggested notebook outline (not executable here)

Notes:
- Place your datasets in a 'data/' folder in the same directory (paths referenced in code).
- The CGM LSTM file has an optional TensorFlow section commented out; install TF if you plan to run it.
- These scripts are intended as a starting point; tune hyperparameters and add logging/experiment tracking for production.
