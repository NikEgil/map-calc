import mapper

heatmap=mapper.HeatMap()
#построить рандомную карту в 
heatmap.example(show=True,info=True,validation=True)

coordinates={}
values={}
borders=[]

geojson=heatmap.map_calc(coordinates, values,borders)
