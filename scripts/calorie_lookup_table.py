import pandas as pd
#https://github.com/syntagmatic/usda-sqlite

foods = ['rgb', 'banana', 'potato', 'kiwi', 'egg', 'tomato']
usda = [np.nan, 'Bananas, raw', 'Potatoes, raw, skin', 'Kiwifruit, green, raw', 'Egg, white, raw, fresh', 'Tomatoes, red, ripe, raw, year round average']

food_to_calorie = dict()

for food in foods:
    food_to_calorie[food] 
print(food_to_calorie)