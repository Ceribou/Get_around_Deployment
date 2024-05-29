import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np
import time

### Config
st.set_page_config(
    page_title="GetAround",
    page_icon="🚗",
    layout="wide"
)

def pie_eda(dataframe, column, title):
    fig = px.pie(dataframe, names=column)
    fig.update_layout(title=title, showlegend=False, title_x=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def bar_eda(dataframe, column, group, title):
    fig = px.bar(dataframe, x = column , color=group)
    fig.update_layout(title=title, title_x=0.3)
    fig.update_xaxes(title=None)
    return fig

def histogram_eda(dataframe, column, title):
    fig = px.histogram(dataframe[column], nbins = 10)
    fig.update_layout(showlegend = False, title = title, title_x=0.3)
    fig.update_xaxes(title=None)
    return fig

def display_probability(dataframe, column, target, title):
    fig = px.histogram(
        dataframe, 
        x=column, 
        color=target, 
        facet_row=target, 
        histnorm='probability',
        text_auto = True
    )
    fig.update_layout(title=title, showlegend=False, title_x=0.1)
    fig.update_xaxes(title=None)
    return fig
    
# Import data
@st.cache_data
def load_data(DATA):
    data = pd.read_csv(DATA)
    return data

df = load_data('df.csv')
pricing = load_data("https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_pricing_project.csv")

df_impact = df.dropna(subset=['previous_ended_rental_id'])
# Drop lines with missing values
df_impact = df_impact.dropna(subset=['previous_delay_in_minutes'])
# Removing outliers (delay over or below 12 hours)
df_impact = df_impact[(df_impact['previous_delay_in_minutes'] <= 720) & (df_impact['previous_delay_in_minutes'] > -720)]
# Adding features about the waiting time of the second user -> if the delay of the first reservation is higher than the time delta,
# it means the second user had to wait for the car
df_impact['waiting_time_second_user'] = df_impact['previous_delay_in_minutes'] - df_impact['time_delta']
# 0 for all lines where the car for ontime for the second user
df_impact['waiting_time_second_user'] = df_impact['waiting_time_second_user'].apply(lambda x: 0 if x <= 0 else x)
# Create a simple column to have the information if the 2nd user had to wait for the car (considering delay as 10 minutes late)
df_impact['waiting_time'] = df_impact['waiting_time_second_user'].apply(lambda x: "yes" if x >= 10 else "no") 


### App
st.title("GetAround analysis 🚗")

st.markdown("""
    Welcome to  dashboard. Here is the analysis of GetAround data.
    
    When using Getaround, drivers book cars for a specific time period, from an hour to a few days long. They are supposed to bring back the car on time, but it happens from time to time that drivers are late for the checkout.

    Late returns at checkout can generate high friction for the next driver if the car was supposed to be rented again on the same day. Customer service often reports users unsatisfied because they had to wait for the car to come back from the previous rental or users that even had to cancel their rental because the car wasn’t returned on time.
    
    The objectives are :
    - Displaying a clear status of the situation ;
    - Understanding causes of late returns ;
    - Proposing recommendations.
""")
## Run the below code if the check is checked ✅
if st.checkbox('Show fleet data'):
    with st.spinner('Wait for it...'):
        time.sleep(1.5)
        st.subheader('Raw data')
        st.write(pricing) 


st.write('')

# First section of the dashboard
st.header("GetAround fleet", divider='gray')


col1, col2 = st.columns(2)

with col1:
    st.metric("Number of cars", len(pricing))
    st.plotly_chart(pie_eda(df, "state", "Proportions of cancelled/ended reservations"), theme='streamlit', use_container_width=False)
    st.write(f"{round((df['state'].value_counts(normalize=True) * 100)[1])}% of reservations are cancelled.")
    
    
with col2:
    st.metric("Number of reservations", len(df))
    st.plotly_chart(histogram_eda(df, "checkin_type", "Proportion of checkin_type"), theme="streamlit", use_container_width=False)
    st.write(f"{round((df['checkin_type'].value_counts(normalize=True) * 100)[0])}% of reservations are checkout with the mobile option which seems to be the main choice of the clients.")


st.write('')
st.write('')
st.write('')
st.markdown("---")

# SECOND SECTION
st.header("Delayed checkouts", divider="gray")

st.markdown("""
    In this analysis, we consider a delayed checkout to be at least 10 minutes after the preset time.
    
""")
st.write('')

# Check differences regarding delays for connect and mobile states
df_connect = df[df['checkin_type'] == "connect"]
df_mobile = df[df['checkin_type'] == "mobile"]

connect_checkout = round(df_connect['delay_in_minutes'].mean())
mobile_checkout = round(df_mobile['delay_in_minutes'].mean())

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(pie_eda(df, "is_late", "Proportion of reservations with delays"), theme="streamlit", use_container_width=True)
    st.write(f"{round((df['is_late'].value_counts(normalize=True) * 100)[1])}% of checkouts are done with delay.")
    st.write('')
    st.metric("Usual delay for connected checkout", f"{connect_checkout} minutes")

with col2:
    st.plotly_chart(bar_eda(df, "checkin_type", "is_late", "Repartition of delays among the checkin types"), theme="streamlit", use_container_width=True)
    st.write('We can see that most of the delayed checkouts depend on mobile choice.')
    st.write('')
    st.metric("Usual delay for mobile checkout", f"{mobile_checkout} minutes")

st.write("Negative values mean that the driver returned the car in advance.")
st.markdown("---")


## THIRD SECTION
# FIRST PART ABOUT CLOSE BOOKING
st.header("Focus on close bookings", divider="gray")
st.markdown("""
    This analysis has been done to show how a reservation can have an impact on the next one, within 12 hours between the first and second.

""")
st.write('')

# Divide data in two datasets (if the 2nd user has to wait for the car or not)
df_delay = df_impact[df_impact['waiting_time'] == "yes"]
df_nodelay = df_impact[df_impact['waiting_time'] == "no"]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Number of relevant reservations", len(df_impact))
    
with col2:
    delayed_impact = round((df_impact['waiting_time'].value_counts(normalize=True) * 100))[1]
    st.metric("Proportion of delayed reservations - due to previous booking", f"{delayed_impact}%")

with col3:
    st.metric("Average waiting time when the first reservation is late", f"{round(df_delay['waiting_time_second_user'].mean())} minutes")

with col4:
    st.metric(f"Time delta between the two reservations when the second is delayed", f"{round(df_delay['time_delta'].mean())} minutes")
    

st.write('')



col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(pie_eda(df_delay, "state", "State for delayed reservations - due to previous booking"), theme="streamlit", use_container_width=True)

with col2:
    st.plotly_chart(pie_eda(df_nodelay, "state", "State for reservations on time - no impact from previous booking"), theme="streamlit", use_container_width=True)
    
st.write('We can notice that there are more cancellations for reservations which are delayed, we can conclude that waiting time has definitly an impact on the state of the reservation.')
st.write('')
st.markdown('---')

# THIRD PART TO CHECK IMPACT OF CHECKIN TYPE
st.subheader("Impact of the checkin type from one reservation on the next")

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(display_probability(df_impact, "previous_checkin_type", "waiting_time", "Has the checkin type of the first reservation an impact on the waiting time of the second user ?"), theme="streamlit", use_container_width=True)

with col2:
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('We can see that there is no signicant difference between the checkin type when there is no waiting time for the client.')
    st.write(f'However, there is a huge impact on delayed reservations. Indeed, 2/3 of delayed reservations are precedeed by a mobile checkout.')
    st.write('On the other hand, only 1/3 of connect checkin type cause waiting time for the next user.')
    st.write('We can conclude that mobile checkin type has a negative impact on next reservations.')
    st.write("Let's see if time delta between two reservations has also an impact on the waiting time.")

st.markdown('---')

# FOURTH PART TO CHECK IMPACT OF TIME DELTA ON WAITING TIME
st.subheader('Focus on time delta')

mean_delta = df_impact.groupby(['time_delta', 'previous_checkin_type'])["waiting_time_second_user"].mean().reset_index()
mean_delta = mean_delta[mean_delta['waiting_time_second_user'] != 0]
mean_delta['waiting_time_second_user'] = round(mean_delta['waiting_time_second_user'])

col1, col2 = st.columns(2)

with col1:
    # Calculate the mean waiting time according to the checkin type chosen by the previous user
    fig1 = px.histogram(mean_delta, x = "time_delta", y = "waiting_time_second_user", color = "previous_checkin_type", histfunc='avg', text_auto=True)
    # fig = px.bar(mean_delta, x = "time_delta", y = "waiting_time_second_user", color = "previous_checkin_type")
    fig1.update_layout(title = "Average waiting time regarding time delta and previous checkin type", title_x=0.2)
    st.plotly_chart(fig1, theme="streamlit", use_container_width=True)
    
with col2:
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write("There are more delays when the time delta between two reservations is below 90 minutes.")
    st.write("The worse combination is : low time delta with mobile checkout for the first reservation -> leads to higher waiting time for the second client.")
    
    
st.markdown('---')
    

# FIFTH PART TO INTRODUCE AN HYPOTHESIS
st.header('Impact of a time delta feature on the waiting time', divider="gray")

st.markdown("""
            The objective is testing a feature which implements :
         - 90 minutes delay after a first reservation with mobile checkout ;
         - 15 minutes delay after a first reservation with connect checkout.
         """)

st.write(f'The feature has been tested on the {len(df_delay)} reservations which were impacted by waiting time for the second client (over 10 minutes).')
st.write('')

df_hypothesis = df_impact.copy()
df_hypothesis['new_time_delta'] = df_hypothesis['time_delta']
df_hypothesis.loc[df_hypothesis['previous_checkin_type'] == 'mobile', 'new_time_delta'] = df_hypothesis.loc[df_hypothesis['previous_checkin_type'] == 'mobile', 'new_time_delta'].apply(lambda x: 90 if x < 90 else x)
df_hypothesis.loc[df_hypothesis['previous_checkin_type'] == 'connect', 'new_time_delta'] = df_hypothesis.loc[df_hypothesis['previous_checkin_type'] == 'connect', 'new_time_delta'].apply(lambda x: 15 if x < 15 else x)

# Calculate the new waiting time 
df_hypothesis['test_waiting_time_second_user'] = df_hypothesis['previous_delay_in_minutes'] - df_hypothesis['new_time_delta']

# 0 for all lines where the car is ontime for the second user
df_hypothesis['test_waiting_time_second_user'] = df_hypothesis['test_waiting_time_second_user'].apply(lambda x: 0 if x <= 0 else x)

# Create a simple column to have the information if the 2nd user had to wait for the car (considering delay as 10 minutes late)
df_hypothesis['test_waiting_time'] = df_hypothesis['test_waiting_time_second_user'].apply(lambda x: "late" if x > 10 else "on time") 


# Calculate proportion of delayed reservations
late_real = round((df_hypothesis['waiting_time'].value_counts(normalize=True)*100)[0])
late_hypothesis = round((df_hypothesis['test_waiting_time'].value_counts(normalize=True)*100)[1])

# New waiting time
df_hypothesis_delay = df_hypothesis[df_hypothesis["waiting_time"] == "yes"]
mean_hypothesis = round(df_hypothesis_delay['test_waiting_time_second_user'].mean())
delay_mean = round(df_delay['waiting_time_second_user'].mean())


col1, col2 = st.columns(2)
with col1 :
    st.subheader("Before feature application")
    st.metric("Proportion of delayed reservations - due to previous booking", f"{delayed_impact}%")
    st.write('')
    st.write('')
    st.metric("Average waiting time for the second client", f"{delay_mean} minutes")
    
with col2 :
    st.subheader("After feature application")
    st.metric("Proportion of delayed reservations - due to previous booking", f"{late_hypothesis}%",
           delta = f"{(late_hypothesis - delayed_impact)/delayed_impact * 100}%")
    st.metric("New average waiting time for the second client", f"{mean_hypothesis} minutes",
          delta= f"{mean_hypothesis - delay_mean} minutes")
    

st.write('')
st.write('')
st.write('')

st.header('Conclusion and recommandations', divider="gray")
st.markdown(f"""
According to the analysis, there are issues due to the time delta between two reservations which is sometimes too low. 
Users may wait for the car up to {delay_mean} minutes when the first client checkout delay is higher than the time delta between the two reservations.
This delay seems to be impacted by the checkin type of the first reservation. 2/3 of reservations with delays are preceeded by a reservation with mobile checkout.  

How long the minimum delay should be for the new feature :
- 90 minutes for cars with mobile checkout
- 15 minutes for cars with connect checkout
- This new feature would have reduced by 50% the number of delayed reservations due to late checkouts from previous bookings.

This new feature would decrease the waiting time from 80 minutes to 5 minutes and would decrease the proportion of delayed reservations (from 10 to 5%)
            """)