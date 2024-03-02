import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the data
df = pd.read_csv("day.csv")
hourly_average_users = pd.read_csv("hourly_average_users.csv")
monthly_average_users = pd.read_csv("monthly_average_users.csv")
workday_average_users = pd.read_csv("workday_average_users.csv")

# Define Streamlit app layout and functionality
st.title("Bike Rental User Analysis")

def home():
    st.header("Welcome to Average User Analysis Dashboard")
    st.write("In here I'm trying to find correlation between number of user based on hour, month, and weekday/end")
    st.write("Use the navigation bar to navigate between data")

def hourly_analysis():

    # AVERAGE USER FOR EACH HOUR
    st.header("1. Average User for Each Hour")

    # Plot 1: Average Casual Users for Each Hour
    st.subheader("Average Casual Users for Each Hour")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(hourly_average_users["hr"], hourly_average_users["casual"], color="#4B8FA9")
    ax1.set_title("Average Casual Users for Each Hour", loc="center", fontsize=20)
    ax1.set_xlabel("Hour", fontsize=12)
    ax1.set_ylabel("Casual Users (Average)", fontsize=12)
    ax1.grid(axis='y')
    st.pyplot(fig1)

    # Plot 2: Average Registered Users for Each Hour
    st.subheader("Average Registered Users for Each Hour")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(hourly_average_users["hr"], hourly_average_users["registered"], color="#4B8FA9")
    ax2.set_title("Average Registered Users for Each Hour", loc="center", fontsize=20)
    ax2.set_xlabel("Hour", fontsize=12)
    ax2.set_ylabel("Registered Users (Average)", fontsize=12)
    ax2.grid(axis='y')
    st.pyplot(fig2)

    # Plot 3: Average Total Users for Each Hour
    st.subheader("Average Total Users for Each Hour")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(hourly_average_users["hr"], hourly_average_users["total"], color="#4B8FA9")
    ax2.set_title("Average Total Users for Each Hour", loc="center", fontsize=20)
    ax2.set_xlabel("Hour", fontsize=12)
    ax2.set_ylabel("Users (Average)", fontsize=12)
    ax2.grid(axis='y')
    st.pyplot(fig2)

def monthly_analysis():
    # AVERAGE USER FOR EACH MONTH
    st.header("2. Average User for Each Month")

    # Plot 1: Average Casual Users for Each Month
    st.subheader("Average Casual Users for Each Month")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(monthly_average_users["mnth"], monthly_average_users["casual"], color="#4B8FA9")
    ax1.set_title("Average Casual Users for Each Month", loc="center", fontsize=20)
    ax1.set_xlabel("Month", fontsize=12)
    ax1.set_ylabel("Casual Users (Average)", fontsize=12)
    ax1.grid(axis='y')
    st.pyplot(fig1)

    # Plot 2: Average Registered Users for Each Month
    st.subheader("Average Registered Users for Each Month")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(monthly_average_users["mnth"], monthly_average_users["registered"], color="#4B8FA9")
    ax2.set_title("Average Registered Users for Each Month", loc="center", fontsize=20)
    ax2.set_xlabel("Month", fontsize=12)
    ax2.set_ylabel("Registered Users (Average)", fontsize=12)
    ax2.grid(axis='y')
    st.pyplot(fig2)

    # Plot 3: Average Total Users for Each Month
    st.subheader("Average Total Users for Each Month")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(monthly_average_users["mnth"], monthly_average_users["total"], color="#4B8FA9")
    ax2.set_title("Average Total Users for Each Month", loc="center", fontsize=20)
    ax2.set_xlabel("Month", fontsize=12)
    ax2.set_ylabel("Users (Average)", fontsize=12)
    ax2.grid(axis='y')
    st.pyplot(fig2)


def weekday_analysis():
    # AVERAGE USER IN WEEKDAY / WEEKEND
    st.header("3. Average User for Each Month")    

    # Plot 3: Average Total Users by Working Day and User Type
    st.subheader("Average User Count by Working Day and User Type")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='workingday', y='value', hue='variable', palette={'casual': 'blue', 'registered': 'green', 'total': 'lightgray'}, data=pd.melt(workday_average_users, id_vars=['workingday'], var_name='variable', value_name='value'), ax=ax3)
    ax3.set_title('Average User Count by Working Day and User Type')
    ax3.set_xlabel('Working Day')
    ax3.set_ylabel('Average User Count')
    ax3.legend(title='User Type')
    st.pyplot(fig3)


def clustering():
    
    # Data Clustering without Machine Learning
    st.header("4. Clustering without Machine Learning")

    # Define segmentation rules based on temperature and humidity
    def segment_temperature_humidity(row):
        if row['temp'] > 0.5 and row['hum'] < 0.6:
            return 'Hot and Dry'
        elif row['temp'] > 0.5 and row['hum'] >= 0.6:
            return 'Hot and Humid'
        elif row['temp'] <= 0.5 and row['hum'] < 0.6:
            return 'Cool and Dry'
        else:
            return 'Cool and Humid'

    # Apply segmentation
    df['clustering_segment'] = df.apply(segment_temperature_humidity, axis=1)
    
    # Scatter plot to visualize clustering
    st.subheader("Scatter Plot: Temperature vs Humidity")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='temp', y='hum', hue='clustering_segment', palette='viridis')
    plt.title("Temperature vs Humidity")
    plt.xlabel("Temperature")
    plt.ylabel("Humidity")
    plt.grid(True)
    st.pyplot()
    
    # Visualize segments
    st.subheader("Temperature and Humidity Count")
    st.bar_chart(df['clustering_segment'].value_counts())

# Run the Streamlit app
if __name__ == '__main__':
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Home", "Hourly Analysis", "Monthly Analysis", "Weekday Analysis", "Clustering"])
    
    if page == "Home":
        home()
    
    elif page == "Hourly Analysis":
        hourly_analysis()
        
    elif page == "Monthly Analysis":
        monthly_analysis()
        
    elif page == "Weekday Analysis":
        weekday_analysis()
        
    elif page == "Clustering":
        clustering()
