# Manufacturing-Decision-Support-System-MDSS-
industry-Standard AI + SPC + SCADA Governance Framework
Process Mean (Average)

Formula in words:
Add all observed process values together, then divide by the total number of observations.

Why it is used:
Represents the normal operating level of the process.

2️ Process Variation (Standard Deviation)

Formula in words:
Measure how far each observation is from the average, square those differences, take their average, and then take the square root.

Why it is used:
Shows how stable or unstable the process is.

3️ Control Limits (Statistical Process Control)
Upper Control Limit

Formula in words:
Take the process average and add three times the process variation.

Lower Control Limit

Formula in words:
Take the process average and subtract three times the process variation.

Why it is used:
Defines the normal operating range of the process.

4️ Z-Score (Process Deviation Level)

Formula in words:
Take the current measured value, subtract the process average, and divide the result by the process variation.

Why it is used:
Shows how abnormal the current value is compared to normal behavior.

5️ Process Health Score (Golden Score)

Formula in words:
Start with a perfect score, then reduce the score as the current process value moves farther away from the average.

Why it is used:
Converts technical variation into a simple percentage health indicator.

6️ Drift Detection Logic

Formula in words:
Continuously store recent process health scores, calculate their average, and flag drift when this average stays below an acceptable threshold for a sustained period.

Why it is used:
Detects slow degradation that traditional alarms miss.

7️ Automation Enable Decision

Formula in words:
Allow automatic control only when the process health score is high, no defects are predicted, and no drift is detected.

Why it is used:
Prevents unsafe automation.

8️ Human Approval Required Decision

Formula in words:
Require operator approval when the process health score is moderate, minor defects are predicted, or early drift signals appear.

Why it is used:
Keeps humans in the loop during uncertainty.

9️ Manual-Only Intervention Decision

Formula in words:
Disable automation immediately when the process health score is low, critical defects are predicted, or strong drift is detected.

Why it is used:
Protects equipment, product quality, and compliance.

10 Defect Risk Estimation

Formula in words:
Use historical defect data and current process conditions to estimate the likelihood of producing a defect in the next production cycle.

Why it is used:
Prevents defects before they occur.

