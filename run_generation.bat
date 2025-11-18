@echo off
SET PYTHONIOENCODING=utf-8
echo.
echo ======================================================================
echo STARTING COMPLETE GA DATASET GENERATION
echo ======================================================================
echo.
echo Configuration:
echo    - 535 solutions (107 apps x 5 seeds)
echo    - 50 generations Global GA
echo    - 30 generations Local GA  
echo    - Estimated time: ~7.7 hours
echo.
echo This window will run overnight. Do not close.
echo ======================================================================
echo.

python generate_all_ga_solutions.py --seeds 5 --timeout 300 --no-skip

echo.
echo ======================================================================
echo GENERATION COMPLETE
echo ======================================================================
echo.
pause
