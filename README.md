# Wavelet Decomposition - Neural Networks (WDNN)
A repository for reproducing the results in the paper *Wavelet Decomposition and Neural Networks—A Potent Combination for Short Term Wind Speed and Power Forecasting*.


**1. Computer and software environment**

Hardware and Operating System:
- Any operating system capable of installing R software and Python (Windows/MacOS/Ubuntu etc.) with a recommended RAM of 16 GB or above.

Software used:
- R version 4.1 [[https://www.R-project.org](https://www.R-project.org/)]
- Python version 3.7 [[https://www.python.org](https://www.python.org/)]

Required R package:
- DSWE version 1.5 or above (Available on CRAN and GitHub) [[https://github.com/TAMU-AML/DSWE-Package](https://github.com/TAMU-AML/DSWE-Package)]

Required Python packages:
- PyWavelets version 1.1 [[https://pywavelets.readthedocs.io/en/latest](https://pywavelets.readthedocs.io/en/latest/)]
- TensorFlow version 2.1 [[https://www.tensorflow.org](https://www.tensorflow.org/)]
- PyTorch version 1.8 [[https://www.pytorch.org](https://www.pytorch.org/)]

**2. Data files**

| **Dataset** | **File names** | **Description** |
| --- | --- | --- |
| #1 | dataset\_1.csv | Dataset-1 containing wind speed, wind direction, first and second derivatives of wind speed, wind power, first and second derivatives of wind power for 36 wind turbines. |
| #2 | dataset\_2.csv | Dataset-2 containing wind speed, first and second derivatives of wind speed, wind power, first and second derivatives of wind power for 160 wind turbines. |
| #3 | dataset\_3.csv | Dataset-3 containing wind speed, first and second derivatives of wind speed for 100 wind turbines. |
| #4 | dataset\_3\_meta.csv | Dataset-3 meta data containing turbine number, site ID, latitude, and longitude for 100 wind turbines. |

**3. Explanation of the headers of the data files**

| **Header name** | **Meaning** |
| --- | --- |
| Time | Timestamp of the data collection. |
| Turbine[t]\_Speed | 10-min average of wind speed observed, in units of m/s. |
| Turbine[t]\_Direction | 10-min average of wind direction observed, in units of degrees. |
| Turbine[t]\_D1Speed | 10-min average of the first derivative of wind speed observed, in units of m/s. |
| Turbine[t]\_D2Speed | 10-min average of the second derivative of wind speed observed, in units of m/s. |
| Turbine[t]\_Power | 10-min average of active power output. |
| Turbine[t]\_D1Power | 10-min average of the first derivative of active power output. |
| Turbine[t]\_D2Power | 10-min average of the second derivative of active power output. |

Where [t] is the number of the turbine. For instance, Turbine5\_Speed is the wind speed measured at Turbine 5.

**4. Reproducing the results in the paper**

| **Code File** | **Results to Reproduce** | **Required Data** | **Output** |
| --- | --- | --- | --- |
| table1.py | Table 1 | dataset\_2.csv | table1a.csv (MAEs)<br/>table1b.csv (wavelet averages)<br/>table1c.csv (k averages)<br/>These files are combined in a spreadsheet to produces Table 1. |
| wdnn\_1\_s\_ffnn.py<br/>wdnn\_1\_s\_rnn.py<br/>wdnn\_1\_s\_lstm.py<br/>table2.py | Table 2 | dataset\_1.csv | table2.csv<br/>Averages are computed on a spreadsheet. |
| wdnn\_[d]\_s\_ffnn.py<br/>wdnn\_[d]\_s\_rnn.py<br/>wdnn\_[d]\_s\_lstm.py<br/>table3.py | Table 3 | dataset\_1.csv<br/>dataset\_2.csv<br/>dataset\_3.csv | table3.csv |
| wdnn\_[d]\_p\_ffnn.py<br/>wdnn\_[d]\_p\_rnn.py<br/>wdnn\_[d]\_p\_lstm.py<br/>table4.py | Table 4 | dataset\_1.csv<br/>dataset\_2.csv | table4.csv |
| table5.py | Table 5 | All files generated while obtaining Tables 3 and 4. | table5.csv |
| per.R<br/>df\_s.pystan\_[d]\_s.py<br/>wdnn\_[d]\_s\_ffnn.py<br/>pstn\_s.py<br/>table6.py | Table 6 | dataset\_1.csv<br/>dataset\_2.csv<br/>dataset\_3.csv<br/>dataset\_3\_meta.csv | table6.csv |
| table7.py | Table 7 | All files generated while obtaining Table 6. | table7.csv |
| table8.py | Table 8 | All files generated while obtaining Table 6. | table8.csv |
| df\_p.R<br/>stan\_p.R<br/>wdnn\_p.R<br/>wdnn\_1\_p\_ffnn.py<br/>wdnn\_2\_p\_ffnn.py<br/>table9.py | Table 9 | dataset\_1.csv<br/>dataset\_2.csv<br/>All files generated while obtaining Table 6. | table9.csv |
| table10.py | Table 10 | All files generated while obtaining Table 9. | table10.csv |

For dataset_1.csv, [t] range: 1 to 36

For dataset_2.csv, [t] range: 1 to 160

For dataset_3.csv, [t] range: 1 to 100

[d] (dataset) range: 1 to 2 for speed and 1 to 3 for power.

[h] (time horizon) range: 1 to 12 unless otherwise specified.

_Note:_

It is strongly advised to execute the code files in the order in which they are presented in the table above.

Other than the files named _table[x].py_ or _table[x].R_, most of the code files require intensive computation (time and resource). In our research, we executed these files using the computing resources of Texas A&M University High Performance Research Computing (HPRC). Some required at least 48 hours to run in the HPRC using 24G memory.
