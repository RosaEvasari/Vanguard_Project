# Vanguard Project

## Project Overview
The digital world is evolving, and so are Vanguard’s clients. Vanguard is US-based investment management company. Vanguard believed that a more intuitive and modern User Interface (UI), coupled with timely in-context prompts (cues, messages, hints, or instructions provided to users directly within the context of their current task or action), could make the online process smoother for clients. The critical question was: **Would these changes encourage more clients to complete the process?**

Vanguard just launched an exciting digital experiment. Our goal as Customer Experience team is helping the company to uncover the results of the experiment. To address the critical question, we have developed a set of hypotheses to guide our analysis and determine whether the new UI and in-context prompts effectively enhance client engagement and completion rates as follows: <br>
1. The new feature would encourage more clients to complete the process. <br>
2. The new feature would reduce the time spent on each step of the process, leading to more efficient completion. <br>
3. The new feature would reduce the error rates during the process, leading to smoother completion. <br>

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

## Experiment Evaluation
As part of the analysis, it's required to carry out experiment evaluation related to design effectiveness and duration of experiment. 

**1. Design Effectiveness** <br>
   It's important to understand whether there is bias in the test and control group for the different variables, such as age, balance, calls_6_month, gender, logons_6_month, number_of_accounts, and tenure_year. We observe that the maximum bias is 4% which appears in the gender categeory. In general, the bias is acceptable for the experiment.

**2. Duration of experiment** <br>
  We also evaluate the number of visits per day for both groups. The result of the analysis shows that the access rate to the page is relatively low during the first two weeks. In addition, the access rate increases significantly after the first two weeks and the access rate in April shows very distinct spikes on the Wednesdays. The high acticity rate in April is probably due to the deadline for filing federal income tax returns in the U.S. Furthermore, the rate decreases in June, but it remains on a relatively steady number that is higher than for the first two weeks.


## Hypotheses and Key Performance Indicator (KPI) 
**1. The new feature would encourage more clients to complete the process** <br>

**KPI**: completion rate during test period.

The completion rate indicates the number of visits which reach 'confirm' step. This analysis takes into account the completion rate per day, which in the end will be concluded by completion rate over the test period. In this case, as shown by the experiment evaluation, the analysis will focus only on the first 55 days.
<br>
Vanguard has set the minimum increase in completion rate at 5%. In other words, if the new design doesn't lead to at least this level of improvement, it may not be justifiable from a cost perspective.

![Completion Rate before 55 days](/resources/charts/Completion_rate_during_test_period_55_days.png)

The result of the analysis shows that the completion rate of the test group is approximately 10% higher than the control group. It means that it has exceeded the threshold set by the company.

**2. The new feature would reduce the time spent on each step of the process, leading to more efficient completion** <br>

**KPI**: Time spent on each step.

In this analysis, we calculate the time spent on each step per visit for both groups. 

The figure below shows that the control group moves slightly faster on average. The total time spent for control and test group is 309.68 s and 315.32 s, respectively.

Furthermore, we analyse the visitor retention per step. The figure below shows that the retention improved from 51.72% to 65.42%. 
- There is significant progress in users retention on 'step_1' and 'confirm'.
- The reduction of time spent on ‘step_1’ significantly reduced % of gone visitors retention (14.7 % compare to 23.8% in control group)

![Completion Rate before 55 days](/resources/charts/time_per_step.png)

The reduction of time spent on ‘step_1’ led to a significant increase of visitors retention (9.1 % in total)
We can recommend to continue to work on retention rate Improvement on ‘step_1’ and ‘step_2’

![Completion Rate before 55 days](/resources/charts/retention_rate_gone_users.png)  

**3. The new feature would reduce the error rates during the process, leading to smoother completion** <br>

**KPI**: Error rates

Every time a user goes a step back, this is counted as an error. The error rate is the number of errors per session. Here the average error rate per day is considered. The figure below shows that there is significant deviation in error rates between test and control group in the first 55 days or 8 weeks. Then, we can see that the error rates between those groups are more comparable.

 <img src="/resources/charts/error_rate_per_day.png" alt="error_rate_per_day" width=500>

Based on this finding, it's decided that the analysis will only take into account the first 55 days or 8 weeks of the trial.

## Conclusion
- The users might need some time to get used to the new layout. In the introduction period, they will make more errors than before.

- The total completion rate of the new feature increased by 10% compared to the old feature exceeding our 5% threshold.

- The reduction of time spent ( on ‘step_1’) led to a significant increase of visitors retention (9.1 % in total) 

## Additional links
- **Notion**: [Project Management in Notion](https://www.notion.so/Tasks-Management-47cc792e7480418bb30dde2c69c34127)
- **Presentation**: [Google Presentation Slides](https://docs.google.com/presentation/d/1G75818Gwe63nyUir6QaTZ9PY0IcC3c65tt_HlW3CBCg/edit#slide=id.g2f01f6d6dde_1_100)
- **PowerBI**: [PowerBI Dashboard](resources/power_bi)




