# -*- coding: utf-8 -*-
"""
Created on Sat May 11 14:05:41 2024

@author: jagad
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'C:\Finlatics_Projects\Global Youtube Statistics.csv', encoding='unicode_escape')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


print(df.describe())
print(df.info())
print(df.nunique())

df = df.drop_duplicates()
print(df.isnull().sum())
df.drop(columns=['Country','Youtuber', 'Abbreviation'], inplace = True)

df.dropna(subset = ['subscribers', 'created_date', 'created_year', 'created_month', 'video_views_rank'], inplace = True)


sns.boxplot(data = df, y = 'Population')
plt.show()
df['Population'] = df['Population'].fillna(df['Population'].median())

sns.boxplot(data = df, y = 'Unemployment rate')
plt.show()
df['Unemployment rate'] = df['Unemployment rate'].fillna(df['Unemployment rate'].mean())

sns.boxplot(data = df, y = 'Urban_population')
plt.show()
df['Urban_population'] = df['Urban_population'].fillna(df['Urban_population'].median())

sns.boxplot(data = df, y = 'Latitude')
plt.show()
df['Latitude'] = df['Latitude'].fillna(df['Latitude'].median())

sns.boxplot(data = df, y = 'Longitude')
plt.show()
df['Longitude']  = df['Longitude'].fillna(df['Longitude'].mean())

sns.boxplot(data = df, y = 'Gross tertiary education enrollment (%)')
plt.show()
df['Gross tertiary education enrollment (%)'] = df['Gross tertiary education enrollment (%)'].fillna(df['Gross tertiary education enrollment (%)'].mean())

sns.boxplot(data = df, y = 'country_rank')
plt.show()
df['country_rank'] = df['country_rank'].fillna(df['country_rank'].median())

sns.boxplot(data = df, y = 'channel_type_rank')
plt.show()
df['channel_type_rank'] = df['channel_type_rank'].fillna(df['channel_type_rank'].median())

sns.boxplot(data = df, y = 'video_views_for_the_last_30_days')
plt.show()
df['video_views_for_the_last_30_days'] = df['video_views_for_the_last_30_days'].fillna(df['video_views_for_the_last_30_days'].median())

sns.boxplot(data = df, y = 'subscribers_for_last_30_days')
plt.show()
df['subscribers_for_last_30_days'] = df['subscribers_for_last_30_days'].fillna(df['subscribers_for_last_30_days'].median())

df['category'] = df['category'].fillna(df['category'].mode()[0])

df['Country of origin'] = df['Country of origin'].fillna(df['Country of origin'].mode()[0])

df['channel_type'] = df['channel_type'].fillna(df['channel_type'].mode()[0])

#answer 1
#Top 10 Youtube channels based on number of subscribers
print("\nAnswer 1")
top_Ten_Youtube_channels = df.sort_values(by=['rank']).head(10)['Title'].tolist()
print(top_Ten_Youtube_channels)

#answer 2
#Category with Highest Average number of subscribers
print("\nAnswer 2")
groupCategory = df.groupby(['category'])
avgTotalSubscribers = groupCategory['subscribers'].mean()
# print(avgTotalSubscribers)
maxAvgSubs = avgTotalSubscribers.max() 
print(avgTotalSubscribers.index[avgTotalSubscribers == maxAvgSubs][0])

    
#answer 3
#Average number of videos uploaded by channels in each categories
print("\nAnswer 3")
groupChannelAndCategory = df.groupby(['Title', 'category'])
print(groupChannelAndCategory['uploads'].mean())

#answer 4
#Top 5 countries with highest number of Youtube channels
print("\nAnswer 4")
groupChannelsByCountry = df.groupby(['Country of origin'])
print(groupChannelsByCountry['Title'].count().sort_values(ascending = False).head(5))

#answer 5
#Distribution of Channel Types across categories 
print("\nAnswer 5")

# Create a countplot
plt.figure(figsize=(40, 20))
sns.countplot(data=df, x='category', hue='channel_type')
plt.title('Distribution of Channel Types Across Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.legend(title='Channel Type')

plt.show()

#answer 6
#correlation between the number of subscribers and total video views for YouTube channels
print("\nAnswer 6")

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='subscribers', y='video views', marker='o', color='b')
plt.title('Correlation Between Subscribers and Total Video Views')
plt.xlabel('Number of Subscribers')
plt.ylabel('Total Video Views')
plt.show()

correlation = df['subscribers'].corr(df['video views'])
print(f"Pearson's correlation coefficient: {correlation:.4f}, shows a HIGH POSITIVE CORRELATION")


#answer 7
#Variation of Monthly earnings through categories
print("\nAnswer 7")
plt.figure(figsize=(25,10)) 
sns.barplot(data = df, x = 'category', y = 'lowest_monthly_earnings')
plt.title("Variation of Monthly earnings through categories")
plt.xlabel('Categories')
plt.ylabel('Lowest Monthly Earnings')
plt.show()


plt.figure(figsize=(25,10)) 
sns.barplot(data = df, x = 'category', y = 'highest_monthly_earnings')
plt.title("Variation of Monthly earnings through categories")
plt.xlabel('Categories')
plt.ylabel('Highest Monthly Earnings')
plt.show()

#answer 8
#overall trend in subscribers gained in the last 30 days across all channels?

print("\nAnswer 8")

plt.figure(figsize=(100,8))
sns.barplot(data = df, x = 'Title', y = 'subscribers_for_last_30_days')
plt.title("trend in subscribers gained in the last 30 days across all channels")
plt.xlabel('Youtube Channels')
plt.ylabel('Subscibers gained in last 30 days')
plt.show()

#answer 9
#Find outliers in yearly earnings from Youtube channels
print("\nAnswer 9")
sns.boxplot(data = df, y = 'lowest_yearly_earnings')
plt.title("box plot for lowest yearly earnings")
plt.show()
sns.boxplot(data = df, y = 'highest_yearly_earnings')
plt.title("box plot for highest yearly earnings")
plt.show()

#answer 10
#distribution of channel creation dates
print("\nAnswer 10")
sns.catplot(data = df, x = 'created_date',hue = 'created_month', col = 'created_year', palette="Set2", kind = 'count')
plt.xlabel('timeline')
plt.ylabel('number of channels')
plt.suptitle("Distribution of channel creation with dates")
plt.show()

#answer 11
#relationship between gross tertiary education enrollment and the number of YouTube channels in a country
print("\nAnswer 11")
groupChannelsByCountryAndEducation = df.groupby(['Country of origin', 'Gross tertiary education enrollment (%)']).count()
#print(groupChannelsByCountryAndEducation)
plt.figure(figsize=(30,15))
sns.scatterplot(data = groupChannelsByCountryAndEducation, x = 'Gross tertiary education enrollment (%)',y = 'Title', hue = 'Country of origin')
plt.xlabel('Gross Tertiary education enrollment (%)')
plt.ylabel('number of youtube channels across countries')
plt.show()


#answer 12
#unemployment rate vary among the top 10 countries with the highest number of YouTube channels?
print("\nanswer 12")
groupbyCountriesWithUnemploymentRate = df.groupby(['Country of origin', 'Unemployment rate']).count().sort_values(by ='rank',ascending = False).head(10)
sns.lineplot(data = groupbyCountriesWithUnemploymentRate, x = 'Unemployment rate', y= 'Country of origin')
plt.xlabel("Unemployment Rate")
plt.ylabel("Number of Youtube channels across top 10 countries")
#print(groupbyCountriesWithUnemploymentRate['Title'])

#answer 13
#average urban population percentage in countries with YouTube channels?

print("\nAnswer 13")
print(df.groupby(['Country of origin'])['Urban_population'].mean())

#answer 14
#distribution of YouTube channels based on latitude and longitude coordinates?

print("\nAnswer 14")
groupByLatLong = df.groupby(['Latitude', 'Longitude']).count()
plt.figure(figsize=(10, 10))
sns.scatterplot(data=groupByLatLong, x='Latitude', y='Longitude',hue = 'Title', marker='o', palette = 'viridis', legend = 'full')
plt.title('distribution of youtube channels across latitudes and longitudes')
plt.xlabel('latitude')
plt.ylabel('longitude')

plt.show()
#print(groupByLatLong['Title'])

#answer 15
#correlation between the number of subscribers and the population of a country

print("\nAnswer 15")

subsInaCountry = df.groupby(['Country of origin','Population']).sum()
#print(subsInaCountry['subscribers'])
plt.figure(figsize=(8, 6))
sns.scatterplot(data=subsInaCountry, x='subscribers', y='Population', marker='o', color='b')
plt.title('correlation between the number of subscribers and the population of a country')
plt.xlabel('number of subscribers across countries')
plt.ylabel('Population across countries')
plt.show()


#answer 16
#top 10 countries with the highest number of YouTube channels compare in terms of their total population

print("\nAnswer 16")
plt.figure(figsize = (15, 10))
groupByCountriesWithHighestChannels = df.groupby(['Country of origin', 'Population']).count().sort_values(ascending = False, by ='rank').head(10)
sns.lineplot(data = groupByCountriesWithHighestChannels, x = 'Country of origin', y ='Population')
plt.xlabel("Countries in order of decreasing number of Youtube channels")
plt.ylabel("Total Population of countries")
plt.show()

#answer 17
#correlation between the number of subscribers gained in the last 30 days and the unemployment rate in a country

print("\nAnswer 17")
plt.figure(figsize = (30, 20))
unEmpRateInCountry = df.groupby(['Country of origin','Unemployment rate']).sum()
sns.scatterplot(data = unEmpRateInCountry, x = 'Unemployment rate', y ='subscribers_for_last_30_days', hue = 'Country of origin')
plt.show()

#answer 18
#distribution of video views for the last 30 days vary across different channel types

print("\nAnswer 18")
plt.figure(figsize=(20,10))
groupedData = df.groupby(['channel_type']).sum()
sns.scatterplot(data = groupedData, x = 'channel_type', y = 'video_views_for_the_last_30_days')
plt.show()



#answer 19
#seasonal trends in the number of videos uploaded by YouTube channels

print("\nAnswer 19")
groupedData = df.groupby(['created_year','created_month']).sum()
sns.catplot(data = groupedData, x= 'created_month', y = 'uploads', col = 'created_year')
plt.xlabel("Months across all years")
plt.ylabel("uploads")
plt.show()































