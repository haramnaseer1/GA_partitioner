@echo off
cd /d "d:\Hira\Freelance\ARman\GA_Partitioner"
python Script\generate_multi_seed_data.py --seeds 1 --apps Application\T2_var_001.json --validate
pause
