@echo off
echo Starting IMM Debug Training with Logging...
echo.

REM Create log directory if it doesn't exist
if not exist "data\logs\debug_perceptual" mkdir "data\logs\debug_perceptual"

REM Generate timestamp for log file
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"
set "timestamp=%YYYY%%MM%%DD%_%HH%%Min%%Sec%"

REM Run training and log to file
echo Log file: data\logs\debug_perceptual\training_log_%timestamp%.txt
echo Start time: %date% %time%
echo.

python debug_train_perceptual.py 2>&1 | tee "data\logs\debug_perceptual\training_log_%timestamp%.txt"

echo.
echo Training completed. Log saved to: data\logs\debug_perceptual\training_log_%timestamp%.txt
pause


