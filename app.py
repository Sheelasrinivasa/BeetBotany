# Imports
import os
from flask import Flask, request, jsonify,render_template
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField

import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, MetaData, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import Column, Integer, String, Numeric, Text, Float, ForeignKey
from sqlalchemy.orm import Session, sessionmaker, relationship

import keras
from keras import backend as K
### END Imports

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'

model = None
graph = None

# Loading a keras model with flask
# https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

total = []

def load_model():
    global model
    global graph
    model = keras.models.load_model("model_trained.h5")
    graph = K.get_session().graph

# Use Pandas to Bulk insert each CSV file into their appropriate table
def populate_table(engine, table, csvfile):
    """Populates a table from a Pandas DataFrame."""
    # connect to the database
    conn = engine.connect()
    
    # Load the CSV file into a pandas dataframe 
    df_of_data_to_insert = pd.read_csv(csvfile)
    
    # Orient='records' creates a list of data to write
    # http://pandas-docs.github.io/pandas-docs-travis/io.html#orient-options
    data = df_of_data_to_insert.to_dict(orient='records')

    # Optional: Delete all rows in the table 
    conn.execute(table.delete())

    # Insert the dataframe into the database in one bulk insert
    conn.execute(table.insert(), data)
    
# Set up database
def database_setup():
    # Create Engine
    engine = create_engine("sqlite:///plantData.sqlite")
    # Declare a Base object here
    Base = declarative_base()

    class PlantData(Base):
        __tablename__ = 'plantData'

        id = Column(Integer, primary_key=True)
        city_name = Column(Text)
        lat = Column(Float)
        lng = Column(Float)
        humidity = Column(Float)
        cloudiness = Column(Float)
        temperature = Column(Float)
        windspeed = Column(Float)
        beets = Column(Text) 
        carrots = Column(Text)
        lettuce = Column(Text)
        parsley = Column(Text)
        radish = Column(Text)
        spinach = Column(Text)
        turnip = Column(Text)

        def __repr__(self):
            return f"id={self.id}, name={self.city_name}"

    # Use `create_all` to create the tables        
    Base.metadata.create_all(engine)

    # Call the function to insert the data for each table
    populate_table(engine, PlantData.__table__, 'PlantData.csv')

    # Use a basic query to validate that the data was inserted correctly for table PlantData`
    print("Retrieving one record from the plantData table", engine.execute("SELECT * FROM plantData LIMIT 1").fetchall())

def query_city_data(city):
    engine = create_engine("sqlite:///plantData.sqlite")
    Base = automap_base()
    Base.prepare(engine, reflect=True)
    plantData = Base.classes.plantData
    session = Session(engine)
    result = session.query(plantData).filter_by(plantData.city_name == city).all()
    # results = session.query(plantData).filter_by(plantData.city_name == city).all()
    beets_data_df = pd.DataFrame(result)

@app.route('/')
def return_homepage():
    return render_template('web/home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_beet_planting():
    # data = {"success": False}
    if request.method == 'POST':
        print(request)
        city_name = request.form['city']
        query_city_data(city_name)

        beetsdata = beets_data_df.drop(["city_ascii","Carrots","Lettuce","Parsley","Radish","Spinach","Turnip"],axis=1)
        beets_X_test = beetsdata.drop("Beets", axis=1)
        prediction = model.predict(beets_X_test)
        return render_template("predict.html", prediction=prediction)
            
    return render_template("predict.html")

if __name__ == "__main__":
    try:
        load_model()
        print("Model loaded")
        database_setup()
        print("Beets Data loaded")

    except Exception as e:
        print("Model loading failed")
        print(str(e))

    app.run(debug=True)
