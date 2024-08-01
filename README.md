# Vanguard Project

## Project Overview
The digital world is evolving, and so are Vanguard’s clients. Vanguard is US-based investment management company. Vanguard believed that a more intuitive and modern User Interface (UI), coupled with timely in-context prompts (cues, messages, hints, or instructions provided to users directly within the context of their current task or action), could make the online process smoother for clients. The critical question was: **Would these changes encourage more clients to complete the process?**

Vanguard just launched an exciting digital experiment. Our goal as Customer Experience team is helping the company to uncover the results of the experiment.

## Data
We use three datasets provided by the company as follows:<br>
1. **Client Profiles** (df_final_demo): Demographics like age, gender, and account details of our clients.<br>
2. **Digital Footprints** (df_final_web_data): A detailed trace of client interactions online.<br>
3. **Experiment Roster** (df_final_experiment_clients): A list revealing which clients were part of the grand experiment.

### Metadata
This comprehensive set of fields will provide you better understandings of the data:
- **client_id**: Every client’s unique ID.
- **variation**: Indicates if a client was part of the experiment.
- **visitor_id**: A unique ID for each client-device combination.
- **visit_id**: A unique ID for each web visit/session.
- **process_step**: Marks each step in the digital process.
- **date_time**: Timestamp of each web activity.
- **tenure_year**: Represents how long the client has been with Vanguard, measured in years.
- **tenure_month**: Further breaks down the client’s tenure with Vanguard in months.
- **age**: Indicates the age of the client.
- **gender**: Specifies the client’s gender.
- **number of accounts**: Denotes the number of accounts the client holds with Vanguard.
- **balance**: Gives the total balance spread across all accounts for a particular client.
- **calls_6_month**: Records the number of times the client reached out over a call in the past six months.
- **logons_6_month**: Reflects the frequency with which the client logged onto Vanguard’s platform over the last six months.

## Key Performance Indicator (KPI) 
- **Completion Rate**: The proportion of users who reach the final ‘confirm’ step.
- **Time Spent on Each Step**: The average duration users spend on each step.
- **Error Rates**: If there’s a step where users go back to a previous step, it may indicate confusion or an error. You should consider moving from a later step to an earlier one as an error.

## Hypotheses
**1. The new feature would encourage more clients to complete the process** <br>
**KPI**: completion rate during test period.

The completion rate indicates the number of visits which reach 'confirm' step. This analysis takes into account the completion rate per day, which in the end will be concluded by completion rate over the test period. In this case, as shown by the experiment evaluation, the analysis will focus only on the first 55 days.
<br>
Vanguard has set the minimum increase in completion rate at 5%. In other words, if the new design doesn't lead to at least this level of improvement, it may not be justifiable from a cost perspective.
<br>
![Completion Rate before 55 days](f'../Completion_rate_during_test_period_55_days.png')

Based on the result, it shows that the completion rate of the test group is approximately 10% higher than the control group. It means that it has exceeded the threshold set by the company.

**2. The new feature would reduce the time spent on each step of the process, leading to more efficient completion** <br>
**KPI**: Time spent on each step.

In this analysis, we calculate the time spent on each step per visit for both groups. 

The figure below shows that the control group moves slightly faster on average. The total time spent for control and test group is 309.68 s and 315.32 s, respectively.

(figure)

Furthermore, we analyse the visitor retention per step. The figure below shows that the retention improved from 51.72% to 65.42%. 
- There is significant progress in users retention on 'step_1' and 'confirm'.
- The reduction of time spent on ‘step_1’ significantly reduced the the drop of visitors retention (14.7 % compare to 23.8% in control group)

(figure)



## Conclusion



## Streamlit App
(link of the app)

## Additional links
- **Notion**:
- **Presentation**:




