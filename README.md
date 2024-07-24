# Vanguard Project

## Project Overview
The digital world is evolving, and so are Vanguard’s clients. Vanguard is US-based investment management company. Vanguard believed that a more intuitive and modern User Interface (UI), coupled with timely in-context prompts (cues, messages, hints, or instructions provided to users directly within the context of their current task or action), could make the online process smoother for clients. The critical question was: **Would these changes encourage more clients to complete the process?**

Vanguard just launched an exciting digital experiment. Our goal as Customer Experience team is helping the company to uncover the results of the experiment.

## Data
We use three datasets provided by the company as follows:<br>
1. Client Profiles (df_final_demo): Demographics like age, gender, and account details of our clients.<br>
2. Digital Footprints (df_final_web_data): A detailed trace of client interactions online.<br>
3. Experiment Roster (df_final_experiment_clients): A list revealing which clients were part of the grand experiment.

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


## Conclusion


## Streamlit App
(link of the app)

## Additional links
### Notion
### Presentation




