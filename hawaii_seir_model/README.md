## Modeling Hawaii Covid data with POMP

### 3/30/2021

I have created 2 directories:
	-test_hawaii
	-test_italy

In the "test_italy" directory there is a script called "italy_test.R" that follows an example inferring transmission dynamics of the infection in italy.
Link to page: https://ionides.github.io/531w20/final_project/Project25/final.html

In the "test_hawaii" directory there is a script called "hawaii_test.R" that attempts to adapt that analysis to the Hawaii data. There is also a "Hawaii_test2.R" that attempts to do with with daily cases instead of total cases. I am currently a little stumped on this one, I cannot read the data into the model (it's probably super easy but I cannot think of it right now).

Feel free to work with them and see what you can add.

-Ethan

###  4/1/21

Timeline of events:
2020:
- March 23: stay at home order
	- effective until May 31
- April 17: state beaches closed
- April 23: face mask mandate
- May 16: Honolulu beaches reopen
- June 5: restaurants re-open for dining in
- June 8: All state beaches reopen
- June 19: Honolulu gyms, recreation areas, bars open
- July 14: Bars/Restaurants stop selling alcohol at midnight
- July 31: Honolulu bars close, restaurants close at 10PM
- August 18: no social gatherings, group limits to 5 or less
- August 27: 2nd stay at home order on O'ahu
	- effective two weeks
- October 15: Safe travels program
- October 22: Honolulu moves to Tier 2 (groups of 5 can be from diff. households, gyms/arcades open at 25% capacity)
- December 15: First vaccine administered
2021:
- February 25: Honolulu moves to Tier 3 (groups of 10 or less)

-Claire

###  4/6/21

I updated the script that Ethan shared last week and broke up the dates into 4 groups (we can always adjust these dates depending on what you guys think):
1. from start of pandemic until May 31 (first lockdown)
June 1, 2020 to July 31, 2020 (end of stay at home order, beaches/restaurants open, close again)
2. August 1, 2020 to October 15, 2020 (Limited social gatherings, second stay at home order on August 27, Safe travels start October 15)
3. October 16, 2020 to December 15, 2020 (Start of Safe travels program to first administered vaccine)
4. December 16, 2020 to March 28, 2021 (First vaccine administration to present)

I'm not sure yet how we can merge these blocks of time, but in the meantime I summed the case counts from the previous time block to the next to reflect the increase in case count. I tried playing around with the parameters as well to line up with the data as best I could, but we should consider what changes make sense based on interventions.

-Claire

### 04/08/21

Nice work Ethan and Claire!! Cool!
Letʻs talk about the breaking up of the data by dates in class today.
 * How is information passed from one date range to the next? Is it via data? Via parameters? Via change in the process?
 * How are the initial parameters chosen for each time segment?

I made one small edit to Claireʻs code. There was no separate rprocess defined for each of the time intervals, only for t1. So I replaced all of them with `covid_rprocess_t1`. Now it works!

Can someone edit the ggplots so that we can see the plots side by side?

-Marguerite

### 04/13/2021

Last week I gathered data on daily COVID-19 vaccine administration in Hawaii. The data begins December 24th. I made modifications to Ethan and Claire's .R code to include vaccines into the model. Vaccines are introduced at the current 'Time 5' in our model. I have not completed the integration of vaccine data into the model. 

I merged the individual time plots into a single ggplot for a better viewing experience! 

-Randi
