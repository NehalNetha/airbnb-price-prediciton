import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

tab1, tab2 = st.tabs(["EDA", "Prediction"])

df = pd.read_csv("final-data-airbnb copy.csv")



mapbox_access_token = "pk.eyJ1IjoibmVoYWwtMzIzIiwiYSI6ImNsb2cwbnR3NDB2NjgyaW4wZXBnNWZyZnMifQ.L3vsORd2xT0upcJT9VtXAA"
px.set_mapbox_access_token(mapbox_access_token)


fig_map = px.scatter_mapbox(df, lat="location/lat", lon="location/lng", color="pricing" ,size="capacity/bedrooms",
                color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10)


fig_state = px.pie(df, names="location/state")

fig_room_type = px.pie(df, names="room_type", hole=0.4)

fig_bed = px.histogram(df, x="room_type", color="capacity/bedrooms", barmode="group")


df["cancel_policy"] = df["cancel_policy"].apply(lambda x: "strict-grace" if x == 'strict_14_with_grace_period'  else x)
df["cancel_policy"] = df["cancel_policy"].apply(lambda x: "super-strict" if x == 'super_strict_30' or x == "super_strict_60" else x)

data_pred = df[['room_type', 'capacity/bathrooms', 'capacity/bedrooms', 'capacity/beds', 'capacity/person', 'reviews/count', 'cancel_policy', 'pricing']]

data_pred['room_type'] = data_pred['room_type'].replace({'entire_home': 0, 'private_room': 1, 'hotel_room': 2, "shared_room": 3})
data_pred['cancel_policy'] = data_pred['cancel_policy'].replace({'firm': 0, 'moderate': 1, 'flexible': 2, "strict-grace": 3, "super-strict" : 4})

X = data_pred[['pricing']]
y = data_pred.drop(['pricing'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)




with tab1:
    st.title("AirBnB Price and Listings Prediction")

    st.subheader("Location of rooms")
    st.text("Plottting Longitudes and Latitues on the map, color saturation for pricing and size of the dots for the total rooms")
    st.plotly_chart(fig_map, theme="streamlit", use_container_width=True)
    st.plotly_chart(fig_state, theme="streamlit", use_container_width=True)

    st.plotly_chart(fig_room_type, theme="streamlit", use_container_width=True)
    st.plotly_chart(fig_bed, theme="streamlit", use_container_width=True)

    option_rel= st.selectbox(
    'Select for histograms of columns',
    ('bathrooms', 'bedrooms', "beds", "People"))

    hist_bath = px.histogram(df, x=  "capacity/bathrooms")
    hist_bed = px.histogram(df, x=  "capacity/bedrooms")
    hist_beds = px.histogram(df, x=  "capacity/beds")
    hist_person = px.histogram(df, x=  "capacity/person")



    if option_rel == 'bathrooms':
        st.plotly_chart(hist_bath, theme="streamlit", use_container_width=True)
    elif option_rel == 'bedrooms':
        st.plotly_chart(hist_bed, theme="streamlit", use_container_width=True)
    elif option_rel == 'Ybeds':
        st.plotly_chart(hist_beds, theme="streamlit", use_container_width=True)
    elif option_rel == 'person':
        st.plotly_chart(hist_person, theme="streamlit", use_container_width=True)




X_pr = data_pred.drop(['pricing', "reviews/count"], axis=1)
y_pr = data_pred[['pricing']]

X_train_pr, X_test_pr, y_train_pr, y_test_pr = train_test_split(X_pr, y_pr, test_size=0.2)
# Create and train a linear regression model
model_price = LinearRegression()
model_price.fit(X_train_pr, y_train_pr)


with tab2:
    st.title("AirBnB Price and Listings Prediction")

    st.title("1. Get better price for your listings")
    st.subheader("If you want to start listing your properties on the airbnb, and want to know what things get that particular pricing?, you can do that by entering the price that you want below")

    input = st.number_input("Enter the Price")
    but_predict = st.button("predict")
    if but_predict:
        prediction = model.predict([[input]])
        rt = round(abs(prediction[0][0]))
        room_type = ""
        if rt == 0:
            room_type = "Entire Room"
        elif rt == 1:
            room_type = "Private Room"
        elif rt == 2:
            room_type = "Hotel Room"
        else:
            room_type = "Shared Room"

        st.subheader(f"Room type should be {room_type}".format(room_type))

        st.subheader(f"The Number of Bathrooms Should be {round(abs(prediction[0][1]))}")

        st.subheader(f"The Number of BedRooms Should be {round(abs(prediction[0][2]))}")

        st.subheader(f"The Number of Beds Should be {round(abs(prediction[0][3]))}")

        st.subheader(f"Capacity for people should be {round(abs(prediction[0][4])) }")



        st.subheader(f"the target reviews on the listing {round(abs(prediction[0][5]))}")

        cp = round(abs(prediction[0][6]))

        cancel = ""
        if cp == 0:
            cancel = "Firm"
        elif cp == 1:
            cancel = "Moderate"
        elif cp == 2:
            cancel = "Flexible"

        elif cp == 3:
            cancel = "Strict Grace"
        else:
            cancel = "Super Strict"




        st.subheader(f"The cancel policy should be {0}".format(cancel))


    st.title("2. Get a Price Prediction")
    st.text("""
        Enter 0: Entire Home
        Enter 1: Private Room
        Enter 2: Hotel Room,
        Enter 3: Shared Room

    """)
    rooms = st.number_input("Enter type of room")
    
    bedrooms = st.number_input("Bedrooms?")
    bathrooms = st.number_input("Bathrooms")
    beds = st.number_input("Beds")
    peoples = st.number_input("People staying?")

    st.text("""
        Enter 0: firm
        Enter 1: moderate
        Enter 2: flexible
        Enter 3: Strict Grace
        Enter 3: Super Strict


    """)
    policy = st.number_input("Cancel Policy?")

    price_predict = st.button("Predict Price")
    if price_predict:
        price_prediction = model_price.predict([[rooms, bedrooms, bathrooms, beds, peoples, policy]])
        st.subheader(round(price_prediction[0][0], 2))






    





