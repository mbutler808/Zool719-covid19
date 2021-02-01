G2 Installation Notes

Following-up with the code. We will be using the file dd_model.py in the main folder (G2_Daily) up through line 423; this is the segment the loads the data, sets parameters, and produces a simple viz of the graphs.

The remainder of the code are extraneous functions used throughout the code for cleaning purposes and visualizations for the site.

---
Bobby just QA'd the code and we ended up removing a line to make it run properly and deleting the code that will not be used for added clarity. You can still share the other code with them as well if they are curious how the visualizations make it to the site, but we won't be executing any of it in class. Lastly, we will have them run the line

'pip install numpy pandas scipy datetime requests bokeh python-dateutil plotly gspread oauth2client'

in the command line prior to running the code to ensure they have all the libraries installed on their computer.

I run all of my code in the IDE Atom (but any will work - Bobby ran on vscode and it executed properly), that way I can execute it line by line or in chunks. If they don't have an IDE set up, they can run ./dd_model.py through the command line. Once they execute through line 421 (show(forecast_graph())) they should have a browser tab pop up with the visualizations in bokeh.

Let me know if you need anything else before the presentation today!
