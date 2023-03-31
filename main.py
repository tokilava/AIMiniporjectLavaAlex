import numpy as np
import pandas as pd
import os


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

#Load the CSV file to read the weather data
weatherData = pd.read_csv('seattle-weather.csv')

#Categorize weather conditions with numbers
"""
 Drizzle = 0
    Rain = 1
     Sun = 2
    Snow = 3
     Fog = 4
"""

weatherData = weatherData.replace({'drizzle': '0', 'rain': '1', 'sun': '2', 'snow': '3', 'fog': '4'})
print(weatherData)

#Check and drop if the weather data has duplicate entries
#weatherData.drop_duplicates()
#print(weatherData)