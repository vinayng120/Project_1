from prophet import Prophet
import pandas as pd
import streamlit as st
import pickle
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.palettes import Category10


def main():
    # Set the app title
    st.title("Oil price prediction using Prophet")

    # Apply custom CSS styling
    st.markdown(
        """
        <style>
            /* Change the font family and color of the heading */
            .title-wrapper {
                font-family: 'Arial', sans-serif;
                color: blue;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Load the data
    df= pd.read_excel("Crude_oil_WTI.xls",parse_dates=True)
    # Renaming the columns
    df.rename(columns={"Cushing, OK WTI Spot Price FOB (Dollars per Barrel)":"price","Date":"date"}, inplace=True)
    Newdf = df.copy()
    Newdf['date'] = pd.to_datetime(Newdf['date'])
    dfm = pd.DataFrame(data=Newdf[['date', 'price']])

    # Add a date input widget
    date_input = st.date_input("Enter a date:")
    predict_button = st.button("Predict", key="predict_button")

    
    if predict_button:
        input_date = pd.to_datetime(date_input)
        
        # Generate predictions for the input date and beyond
        loadmodel=pickle.load(open('model.pkl','rb'))
        future_dates = pd.date_range(start=dfm['date'].min(), end=input_date, freq='D')
        predictions = loadmodel.predict(pd.DataFrame({'ds': future_dates}))
        
        # Get the forecasted price for the input date
        forecasted_price = predictions.loc[predictions['ds'] == input_date, 'yhat'].values
        if len(forecasted_price) > 0:
            forecasted_price = forecasted_price[0]
            st.write("Forecasted Oil Price (USD/BBL):", forecasted_price)
        else:
            st.write("No forecast available for the specified date.")

       

        # Visualize the graph
        st.subheader("Oil Price Prediction-Prophet model")
        p = figure(x_axis_type='datetime', title='Oil Price Prediction-Prophet model', width=800, height=400)
        p.line(dfm['date'], dfm['price'], line_color='green', legend_label='Actual Price')
        p.line(predictions['ds'], predictions['yhat'], line_color='orange', legend_label='Forecasted Price')
        
          

        #Create DataFrame for hovertool
        hover_df = pd.DataFrame({'date': predictions['ds'], 'price': predictions['yhat']})
        # Create a ColumnDataSource for the hover tool
        source = ColumnDataSource(hover_df)
        
        hover_df1 = pd.DataFrame({'date': dfm['date'], 'price': dfm['price']})
         # Update hover_df with actual values
        source1 = ColumnDataSource(hover_df1)
        
        # Add hover tool
        hover_tool = HoverTool(tooltips=[
            ('Date', '@date{%F}'),
            ('Price', '@price{0.00}')
        ], formatters={'@date': 'datetime'})

        p.add_tools(hover_tool)
        p.circle('date', 'price', size=4, fill_color=Category10[3][0], source=source)
        p.circle('date', 'price', size=1, fill_color=Category10[3][0], source=source1)

         
        # Style the plot
        p.xaxis.axis_label = 'Date'
        p.yaxis.axis_label = 'Price (USD/BBL)'
        p.legend.location = 'top_left'
        p.legend.title = 'Legend'
        #p.legend.location = 'top_left'
        #p.legend.click_policy = 'hide'


        # Display the graph
        st.bokeh_chart(p)


if __name__ == '__main__':
    main()
