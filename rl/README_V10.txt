CHAMPION V10
============

Copy the rl/*.py files into:
Stock Assistant\rl\

Compilation:
-----------
python -m py_compile .\rl\champion_v10_features.py
python -m py_compile .\rl\champion_v10_env.py
python -m py_compile .\rl\multi_stock_champion_v10.py
python -m py_compile .\rl\train_champion_v10.py
python -m py_compile .\rl\evaluate_champion_v10.py

Training:
---------
python .\rl\train_champion_v10.py

Evaluation:
-----------
python .\rl\evaluate_champion_v10.py

IMPORTANT
---------
Do NOT deploy to OCI unless the evaluation explicitly says:

STATUS: CHAMPION PASSED

This is an experimental research model, not a guarantee of profitable trading.
