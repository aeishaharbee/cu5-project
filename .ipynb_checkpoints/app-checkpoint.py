import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

pipeline = joblib.load('hotel_price_prediction_model.pkl')

st.set_page_config(page_title="Hotel Price Prediction", layout="wide")
st.sidebar.title("Hotel Price Prediction")
page = st.sidebar.selectbox("", ["Home", "Data Visualization", "Predict"])
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

df = pd.read_csv('cleaned_hotels_data.csv')

# Home Page
if page == "Home":
    st.title("Hotel Price Prediction App")
    st.write("""
        Welcome to the Hotel Price Prediction App. 
        Use this tool to:
        - Visualize data trends in hotel pricing.
        - Predict hotel prices based on various features.
        Navigate through the options in the sidebar to explore.
    """)

# Data Visualization Page
elif page == "Data Visualization":
    st.title("Data Visualization")
    st.write("Here are some of the visualizations of the hotel price trends and patterns.")
    st.markdown("<br><br>", unsafe_allow_html=True)

    # first row
    col1, col2 = st.columns((2))
    avg_accom = df.groupby('accommodation_type')['price_per_night'].mean().sort_values(ascending=False).reset_index()
    avg_city = df.groupby('city')['price_per_night'].mean().sort_values(ascending=False).reset_index()
    
    with col1:
        st.subheader("Average Price by Accommodation Type")
        fig = px.bar(avg_accom, x = "accommodation_type", y = "price_per_night",
                     template = "seaborn")
        fig.update_layout(
            xaxis_title="Accommodation Type", 
            yaxis_title="Average Price Per Night (RM)", 
        )
        st.plotly_chart(fig,use_container_width=True, height = 200)
    
    with col2:
        st.subheader("Average Price by City")
        fig = px.bar(avg_city, x = "city", y = "price_per_night",
                     template = "seaborn")
        fig.update_layout(
            xaxis_title="City", 
            yaxis_title="Average Price Per Night (RM)",
            xaxis=dict(
                tickangle=45  # Rotate x-axis labels by 45 degrees
            ),
        )
        st.plotly_chart(fig,use_container_width=True, height = 200)
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)

    # second row
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.subheader("Time Series Visualizations")
    with col2:
        day_of_month_button = st.button("Day of Month")
    with col3:
        day_of_week_button = st.button("Day of Week")
    with col4:
        month_button = st.button("Month")
    
    time_series_option = "Day of Month" 
    
    if day_of_week_button:
        time_series_option = "Day of Week"
    elif month_button:
        time_series_option = "Month"
    
    if time_series_option == "Day of Month":
        time_series_data = df.groupby('day_of_month')['price_per_night'].mean().reset_index()
        x_axis = 'day_of_month'
        x_title = "Day of the Month"
    elif time_series_option == "Day of Week":
        time_series_data = df.groupby('day_of_week')['price_per_night'].mean().sort_values(ascending=False).reset_index()
        x_axis = 'day_of_week'
        x_title = "Day of the Week"
    else: 
        time_series_data = df.groupby('month')['price_per_night'].mean().reset_index()
        x_axis = 'month'
        x_title = "Month"
    
    fig = px.line(time_series_data, x=x_axis, y="price_per_night", template="seaborn")
    fig.update_layout(
        title=f"Average Price Per Night vs {x_title}",
        xaxis_title=x_title,
        yaxis_title="Average Price Per Night ($)",
    )
    fig.update_traces(mode="lines+markers")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)

    # third row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("User Average Rating vs Price Per Night")
        fig1 = px.scatter(df, x='user_average_rating', y='price_per_night', 
                          labels={'user_average_rating': 'User Average Rating', 
                                  'price_per_night': 'Price Per Night'})
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Hotel Rating vs Price Per Night")
        fig2 = px.scatter(df, x='hotel_rating', y='price_per_night', 
                          labels={'hotel_rating': 'Hotel Rating', 
                                  'price_per_night': 'Price Per Night'})
        st.plotly_chart(fig2, use_container_width=True)

# Predict Page
elif page == "Predict":
    if "input_data" not in st.session_state:
        st.session_state.input_data = {
            'accommodation_type': 'Hotel',
            'hotel_rating': 4,
            'user_average_rating': 8.5,
            'user_comment': 'Good',
            'city': 'Georgetown',
            'rooms': 1,
            'guests': 2,
            'free_wifi': 1,
            'pool': 1,
            'parking': 0,
            'aircond': 1,
            'restaurant': 1,
            'hotel_bar': 1,
            'balcony_patio': 0,
            'spa': 0,
            'gym': 0,
            'hairdryer': 1,
            'kitchen': 0,
            'non_smoking_room': 1,
            'tv_ent': 1,
            'washing_machine': 0,
            'holiday': 0,
            'month': 12,
            'day': 28,
            'day_of_week': 5
        }
    
    example_1 = {
        'accommodation_type': 'Hotel',
        'hotel_rating': 4,
        'user_average_rating': 8.5,
        'user_comment': 'Good',
        'city': 'Georgetown',
        'rooms': 1,
        'guests': 2,
        'free_wifi': 1,
        'pool': 1,
        'parking': 0,
        'aircond': 1,
        'restaurant': 1,
        'hotel_bar': 1,
        'balcony_patio': 0,
        'spa': 0,
        'gym': 0,
        'hairdryer': 1,
        'kitchen': 0,
        'non_smoking_room': 1,
        'tv_ent': 1,
        'washing_machine': 0,
        'holiday': 0,
        'month': 12,
        'day': 28,
        'day_of_week': 5
    }
    
    example_2 = {
        'accommodation_type': 'Resort',
        'hotel_rating': 3,
        'user_average_rating': 7.0,
        'user_comment': 'Excellent',
        'city': 'Kuala Lumpur',
        'rooms': 2,
        'guests': 4,
        'free_wifi': 1,
        'pool': 0,
        'parking': 1,
        'aircond': 1,
        'restaurant': 0,
        'hotel_bar': 0,
        'balcony_patio': 1,
        'spa': 0,
        'gym': 0,
        'hairdryer': 1,
        'kitchen': 1,
        'non_smoking_room': 1,
        'tv_ent': 1,
        'washing_machine': 1,
        'holiday': 1,
        'month': 5,
        'day': 12,
        'day_of_week': 3
    }
    
    st.title("Predict Hotel Price")
    st.write("Predict hotel prices using our trained model. Choose an example or enter custom inputs.")
    
    if st.button("Use Example 1"):
        st.session_state.input_data = example_1
        st.rerun()
    
    if st.button("Use Example 2"):
        st.session_state.input_data = example_2
        st.rerun()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.session_state.input_data['accommodation_type'] = st.selectbox(
            "Accommodation Type",
            ['Hotel', 'Entire House / Apartment', 'Serviced apartment', 'Guesthouse', 'Motel', 'Hostel', 'Resort', 'Bed & Breakfast', 'Camping site'],
            index=['Hotel', 'Entire House / Apartment', 'Serviced apartment', 'Guesthouse', 'Motel', 'Hostel', 'Resort', 'Bed & Breakfast', 'Camping site'].index(st.session_state.input_data['accommodation_type'])
        )
        st.session_state.input_data['hotel_rating'] = st.selectbox(
            "Hotel Rating (1-5)", [1, 2, 3, 4, 5], index=st.session_state.input_data['hotel_rating'] - 1
        )
        st.session_state.input_data['user_average_rating'] = st.slider(
            "User Average Rating (0-10)", 0.0, 10.0, st.session_state.input_data['user_average_rating']
        )
        st.session_state.input_data['user_comment'] = st.selectbox(
            "User Comment", ['Excellent', 'Very good', 'Good', 'Okay', 'Fair'], index=['Excellent', 'Very good', 'Good', 'Okay', 'Fair'].index(st.session_state.input_data['user_comment'])
        )
        st.session_state.input_data['city'] = st.selectbox(
            "City", ['Kuala Lumpur', 'Malacca', 'Kota Kinabalu', 'Genting Highlands', 'Johor Bahru', 'Ipoh', 'Kuantan', 'Kuching', 'Georgetown', 'Putrajaya'],
            index=['Kuala Lumpur', 'Malacca', 'Kota Kinabalu', 'Genting Highlands', 'Johor Bahru', 'Ipoh', 'Kuantan', 'Kuching', 'Georgetown', 'Putrajaya'].index(st.session_state.input_data['city'])
        )
        st.session_state.input_data['rooms'] = st.number_input("Rooms", min_value=1, value=st.session_state.input_data['rooms'])
        st.session_state.input_data['guests'] = st.number_input("Guests", min_value=1, value=st.session_state.input_data['guests'])
    
    with col2:
        for key in ['free_wifi', 'pool', 'parking', 'aircond', 'restaurant', 'hotel_bar', 'balcony_patio', 'spa', 'gym', 'hairdryer', 'kitchen']:
            current_value = st.session_state.input_data[key]
            st.session_state.input_data[key] = st.selectbox(
                f"{key.replace('_', ' ').title()} (1=Yes, 0=No)",
                [1, 0],
                index=[1, 0].index(current_value)  # Explicitly calculate index
            )
    
    with col3:
        for key in ['non_smoking_room', 'tv_ent', 'washing_machine', 'holiday']:
            current_value = st.session_state.input_data[key]
            st.session_state.input_data[key] = st.selectbox(
                f"{key.replace('_', ' ').title()} (1=Yes, 0=No)",
                [1, 0],
                index=[1, 0].index(current_value)  # Explicitly calculate index
            )

            
        st.session_state.input_data['month'] = st.number_input("Month", min_value=1, max_value=12, value=st.session_state.input_data['month'])
        st.session_state.input_data['day'] = st.number_input("Day", min_value=1, max_value=31, value=st.session_state.input_data['day'])
        st.session_state.input_data['day_of_week'] = st.number_input("Day of Week (0=Monday)", min_value=0, max_value=6, value=st.session_state.input_data['day_of_week'])
    
    input_data_df = pd.DataFrame([st.session_state.input_data])
    
    if st.button("Predict"):
        prediction = pipeline.predict(input_data_df)
        st.write(f"Predicted Price per Night: RM {prediction[0]:.2f}")
