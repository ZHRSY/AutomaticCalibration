
from osgeo import ogr,osr
from math import ceil
import os,json,time 
from shapely.geometry import Polygon,Point
from shapely import wkt,ops
import numpy as np
from typing import Tuple,IO

from raindata import RainData

outdir = "./output_meshrain/"
ogr.UseExceptions()

def OpenLog(logfile:str,mode:str):
    if logfile == None:
        return None
    elif isinstance(logfile,str):
        fw = open(logfile,mode)
        return fw
    else:
        raise ValueError("not a valid input")

def WriteLog(fw:IO,string:str):
    if fw == None:
        pass
    else:
        fw.write(string)
def CloseLog(fw:IO):
    if fw == None:
        pass
    else:
        fw.close()



def ReadBasin(basinshp):
    driver = ogr.GetDriverByName('ESRI Shapefile') #shp驱动器
    datasource = driver.Open(basinshp, 0) #打开流域shp文件
    layer = datasource.GetLayer(0) #shp图层默认为0
    Extent = layer.GetExtent() #(xmin,xmax,ymin,ymax)
    EPSG = layer.GetSpatialRef().GetAttrValue('AUTHORITY',1)
    # print('shp:%s'%basinshp)
    # print("EPSG:%s"%EPSG)
    feature = layer[0] #要素
    geom = feature.GetGeometryRef() #要素几何
    basin_poly = wkt.loads( geom.ExportToWkt() ) #根据wkt生成多边形对象
    datasource.Destroy()
    return basin_poly,EPSG,Extent

def ReadGrid(gridshp):
    driver = ogr.GetDriverByName('ESRI Shapefile') #shp驱动器
    datasource = driver.Open(gridshp, 0) #0为只读，1为可写：
    layer = datasource.GetLayer(0)
    Extent = layer.GetExtent() #(xmin,xmax,ymin,ymax)
    EPSG = layer.GetSpatialRef().GetAttrValue('AUTHORITY',1)
    # print('shp:%s'%gridshp)
    # print("EPSG:%s"%EPSG)
    num_grid = len(layer)
    GridPolys = [[]] * num_grid
    for i in range(num_grid):
        feature = layer[i] #网格要素
        geom = feature.GetGeometryRef() #网格要素的几何
        grid_poly = wkt.loads( geom.ExportToWkt() )
        GridPolys[i] = grid_poly
    datasource.Destroy()
    return GridPolys,EPSG,Extent

def ReadTable(shp,name,sort=None):
    driver = ogr.GetDriverByName('ESRI Shapefile') #shp驱动器
    datasource = driver.Open(shp, 0) #0为只读，1为可写：
    layer = datasource.GetLayer(0)
    fields = layer.schema
    values = []
    for feat in layer:
        value = feat.GetField(name)
        if str(type(value)) == "<class 'str'>":
            values.append(value.strip())
        else:
            values.append(value)
    
    if sort == None:
        datasource.Destroy()
        return values
    else:
        sort_values = []
        for feat in layer:
            sort_values.append( feat.GetField(sort) )
        sort_values = np.array(sort_values,dtype=float)
        index_sorted = sort_values.argsort()
        values = np.array(values)
        values = values[index_sorted]

        datasource.Destroy()
        return list(values),index_sorted

def ReadPolys(shp):
    driver = ogr.GetDriverByName('ESRI Shapefile') #shp驱动器
    datasource = driver.Open(shp, 0) #0为只读，1为可写：
    layer = datasource.GetLayer(0)
    Extent = layer.GetExtent() #(xmin,xmax,ymin,ymax)
    EPSG = layer.GetSpatialRef().GetAttrValue('AUTHORITY',1)
    # print('shp:%s'%shp)
    # print("EPSG:%s"%EPSG)
    num_polys = len(layer)
    Polys = [[]] * num_polys
    for i in range(num_polys):
        feature = layer[i] #网格要素
        geom = feature.GetGeometryRef() #网格要素的几何
        polywkt = geom.ExportToWkt()
        if "MULTIPOLYGON" in polywkt:
            num = geom.GetGeometryCount()
            length = np.zeros(num)
            g_list = []
            for iter in range(num):
                g = geom.GetGeometryRef(iter)
                g_list.append( g )
                length[iter] = len(g.ExportToWkt())
            geom = g_list[np.argmax(length)]
            polywkt = geom.ExportToWkt()
            poly = wkt.loads( polywkt )
            Polys[i] = poly
            continue    
        poly = wkt.loads( polywkt )
        Polys[i] = poly
    datasource.Destroy()
    return Polys,EPSG,Extent

def Fishgrid(outfile,EPSG,
             xmin,xmax,ymin,ymax,
             gridwidth,gridheight):
    #参数转换到浮点型
    xmin = float(xmin)
    xmax = float(xmax)
    ymin = float(ymin)
    ymax = float(ymax)
    gridwidth = float(gridwidth)
    gridheight = float(gridheight)

    #计算行数和列数
    rows = ceil((ymax-ymin)/gridheight)
    cols = ceil((xmax-xmin)/gridwidth)

    #初始化起始格网四角范围
    ringXleftOrigin = xmin
    ringXrightOrigin = xmin+gridwidth
    ringYtopOrigin = ymax
    ringYbottomOrigin = ymax-gridheight

    #创建输出文件
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG)
    outdriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outfile):
        outdriver.DeleteDataSource(outfile)
    outds = outdriver.CreateDataSource(outfile)
    outlayer = outds.CreateLayer(outfile,srs,geom_type = ogr.wkbPolygon)
    #不添加属性信息，获取图层属性
    outfielddefn  = outlayer.GetLayerDefn()
    #遍历列，每一列写入格网
    col = 0
    while col<cols:
        #初始化，每一列写入完成都把上下范围初始化
        ringYtop = ringYtopOrigin
        ringYbottom = ringYbottomOrigin
        #遍历行，对这一列每一行格子创建和写入
        row = 0
        while row<rows:
            #创建左上角第一个格子
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(ringXleftOrigin,ringYtop)
            ring.AddPoint(ringXrightOrigin,ringYtop)
            ring.AddPoint(ringXrightOrigin,ringYbottom)
            ring.AddPoint(ringXleftOrigin,ringYbottom)
            ring.CloseRings()
            #写入几何多边形
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
            #创建要素，写入多边形
            outfeat = ogr.Feature(outfielddefn)
            outfeat.SetGeometry(poly)
            #写入图层
            outlayer.CreateFeature(outfeat)
            outfeat = None
            #下一多边形，更新上下范围
            row+=1
            ringYtop = ringYtop - gridheight
            ringYbottom = ringYbottom-gridheight
        #一列写入完成后，下一列，更新左右范围
        col+=1
        ringXleftOrigin = ringXleftOrigin+gridwidth
        ringXrightOrigin = ringXrightOrigin+gridwidth
    #写入后清除缓存
    outds.Destroy()

def extra_node(feature):
    """
    获取多边形节点坐标，存储为二维列表，o为x坐标，1为有坐标
    :param: eature，获取的多边形:
    :return: 多边形节点坐标对，列表
    """
    list = []
    geom = feature.GetGeometryRef() #读取到多边形
    polygon = geom.GetGeometryRef(0) #读取到多边形环
    for i in range(polygon.GetPointCount()):
        x = polygon.GetX(i)
        y = polygon.GetY(i)
        list.append([x, y])
    return list
    
def Intersection_analysis(gridshp,basinshp,debug=False):
    '''
    gridshp must be shpfile, not list of grids, in order to add things in the shp's table
    '''
    # print("performing intersction analysis ...")
    # 读取流域shp文件
    if str(type(basinshp)) == "<class 'str'>":
        basin_poly = ReadBasin(basinshp)[0]  
    elif str(type(basinshp)) == "<class 'shapely.geometry.polygon.Polygon'>":
        basin_poly = basinshp
    else:
        raise TypeError("Wrong type of basinshp in Intersection_analysis!")
    
    # 读取网格shp文件
    GriPolys = ReadGrid(gridshp)[0]  
    num_grid   = len(GriPolys)

    # 计算重叠区面积、网格面积和重叠区占网格面积的比
    coverArea = np.ones(num_grid)
    gridArea  = np.ones(num_grid)
    for i in range(num_grid):
        if GriPolys[i].intersects(basin_poly):
            coverArea[i] = GriPolys[i].intersection(basin_poly).area / (1e+6)
        else:
            coverArea[i] = 0.0
        gridArea[i]  = GriPolys[i].area / (1e+6)
        coverArea[i] = round(coverArea[i],4)
        gridArea[i]  = round(gridArea[i] ,4)
    ratioArea = coverArea/gridArea #重叠区占网格面积的比

    # 在网格shp文件里添加新字段
    if debug:
        newField = {'cover_area': coverArea,
                    'gridArea'  : gridArea,
                    'ratio_area': ratioArea}
        Add_field(gridshp,newField)
    
    # print("="*100)
    return coverArea

def Add_field(shp,newField):
    # print("adding/updating fields ...")
    driver = ogr.GetDriverByName('ESRI Shapefile') #shp驱动器
    datasource = driver.Open(shp, 1) #打开流域shp文件
    layer = datasource.GetLayer(0) #shp图层默认为0
    num_features = len(layer) #要素个数

    # 打印已有字段
    exist_Fields = [field.name for field in layer.schema]
    # print('Before: the shp file has fields below:')
    # print(exist_Fields)
    
    # 检查newField字典中的每个key的值的元素个数是否等于shp文件中的要素个数
    Fieldnames = newField.keys() #新字段列表
    num_name = len(Fieldnames)
    list_num = []
    for name in Fieldnames:
        num_new = len(newField[name])
        list_num.append(num_new)
    if sum(abs(list_num - num_features*np.ones(num_name))) == 0:
        pass
    else:
        # print(list_num)
        raise ValueError("The numbers of element in newField are not equal to that of features.")

    # 检查是否为已有字段，若否添加新字段
    for name in Fieldnames:
        if name not in exist_Fields:
            field = ogr.FieldDefn(name,ogr.OFTReal)
            #field.SetWidth(20) 
            #field.SetPrecision(3)
            layer.CreateField(field)
            # print('creating field %s ...'%name)
        else:
            # print('updating field %s ...'%name)
            pass

    for i in range(num_features):
        feature = layer[i]
        for name in Fieldnames:
            feature.SetField(name, newField[name][i])
        layer.SetFeature(feature) #更新要素  

    # 打印添加后的已有字段
    exist_Fields = [field.name for field in layer.schema]
    # print('After: the shp file has fields below:')
    # print(exist_Fields)

    datasource.Destroy()
    # print("="*100)
    return
    
def points_in_poly(points, polyline_points):
    """
    说明：获取多边形图层内部的网格节点，返回值为布尔数组\n
    变量：
     - @points (list-like)          点列表
     - @polyline_points (list-like) 多边形顶点按顺序组成的点列表
    """
    from matplotlib.path import Path
    poly_path = Path(polyline_points)
    inside_poly = poly_path.contains_points(points)
    return inside_poly

def Get_Grid_points(gridshp,output=None,isdensify=None):
    # print("geting grid points ...")
    GridPolys,EPSG = ReadGrid(gridshp)[:2]
    Grid_points = []
    for poly in GridPolys:
        xy = poly.exterior.coords[:] #多边形需先提取外环，再返回坐标
        for item in xy:
            if item not in Grid_points:
                Grid_points.append(item)
    Grid_points = np.array(Grid_points)
    
    if isdensify != None:
        add_points = []
        for grid in GridPolys:
            for layer in isdensify:
                if abs(layer - 0) < 1e-3:
                    center = np.array(grid.centroid.coords[:][0])
                    add_points.append( center )
                elif (layer > 0) and (layer < 1):
                    gridpoints = np.array( grid.exterior.coords[:][0:-1] )
                    center = np.array( grid.centroid.coords[:][0] )
                    points = (gridpoints - center)*layer + center
                    for item in points: add_points.append(item)

        add_points = np.array(add_points)
        Grid_points = np.vstack((Grid_points,add_points))

    if output != None:
        if EPSG == '4326':
            np.savetxt(output,Grid_points,delimiter=',',fmt='%.6f',encoding="utf-8")
        else:
            np.savetxt(output,Grid_points,delimiter=',',fmt='%.2f',encoding="utf-8")

    # print("="*100)
    return Grid_points

def Get_BasinCenter(basinshp,output=None):
    # print("geting centroids of basins ...")
    BasinPolys,EPSG = ReadPolys(basinshp)[:2]
    centroids = []
    for basin in BasinPolys:
        centroids.append( basin.centroid.coords[:][0] )
    centroids = np.array(centroids)

    if output != None:
        if EPSG == '4326':
            np.savetxt(output,centroids,delimiter=',',fmt='%.6f',encoding="utf-8")
        else:
            np.savetxt(output,centroids,delimiter=',',fmt='%.2f',encoding="utf-8")

    # print("="*100)
    return centroids




def Read_discdata(jsonfile):
    # print("reading rain data ...")
    
    Qs  = []
    with open(jsonfile,'r',encoding="utf-8") as fr:
        js = json.load(fr)['Flowrate']
        
        for iter,item in enumerate(js):
            Q = []
            T_series = []
            for jtem in item['Qdatas']:
                Q.append(jtem['value'])
                T_series.append(jtem['time'])
            Qs.append(Q)
    T_series = np.array(T_series,dtype=str)
    Qs  = np.array(Qs,dtype=float)
    return T_series,Qs

def Read_raindata(jsonfile):
    # print("reading rain data ...")
    
    RDpoints = []
    RDrains  = []
    num_data = []
    station_names = []
    with open(jsonfile,'r',encoding="utf-8") as fr:
        js = json.load(fr)['rainMeshs']
        
        for iter,item in enumerate(js):
            if 'name' in item:
                station_names.append(item['name'])

            RDpoints.append([item['lon'],item['lat']])
            rain = []
            T_series = []
            for jtem in item['meshedatas']:
                rain.append(jtem['value'])
                T_series.append(jtem['time'])
            RDrains.append(rain)
            num_data.append( len(rain) )

    mode = np.bincount(num_data).argmax() #众数
    for i,item in enumerate(num_data):
        if item != mode:
            print(f"Error:\n  point:{RDpoints[i]} | number of data:{item}, expected {mode}")

    RDpoints = np.array(RDpoints,dtype=float)
    RDrains  = np.array(RDrains,dtype=float)
    T_series = np.array(T_series,dtype=str)
    raindata = RainData(RDrains,RDpoints)
    raindata.Tseries = T_series
    if len(station_names) != 0:
        raindata.points_name = np.array(station_names,dtype=str)
    raindata.EPSG = 4326
    
    # print("raindata: %d grid points with %d values in terms of time."
    #      %(np.size(RDrains,0),np.size(RDrains,1)))
    # print("="*100)
    return raindata




def Read_raindata_basin_dependent(jsonfile):
    # print("reading rain data ...")
    
    with open(jsonfile,'r',encoding="utf-8") as fr:
        js = json.load(fr)['rainMeshs']

    RDpoints = []
    RDrains  = []
    basin_names = []
    station_names = []
    for iter,item in enumerate(js):
        basin_names.append(item['basin'])
        station_names.append(item['name'])

        RDpoints.append([item['lon'],item['lat']])
        rain = []
        T_series = []
        for jtem in item['meshedatas']:
            rain.append(jtem['value'])
            T_series.append(jtem['time'])
        RDrains.append(rain)
    RDpoints = np.array(RDpoints,dtype=float)
    RDrains  = np.array(RDrains,dtype=float)
    T_series = np.array(T_series,dtype=str)
    
    basin_names_dif = []
    for item in basin_names:
        if item not in basin_names_dif: basin_names_dif.append(item)
    basin_names   = np.array(basin_names,dtype=str)
    station_names = np.array(station_names,dtype=str)

    AllData = {}
    for basin in basin_names_dif:
        AllData[basin] = {}
        bool = basin_names == basin

        AllData[basin] = RainData(RDrains[bool], RDpoints[bool])
        AllData[basin].points_name = station_names[bool]
        AllData[basin].Tseries = T_series
        AllData[basin].EPSG = 4326
        
    # print("raindata: %d grid points with %d values in terms of time."
    #      %(np.size(RDrains,0),np.size(RDrains,1)))
    # print("="*100)
    return AllData


def Generate_example_jsonfile(rains:np.ndarray,
                              points:np.ndarray,
                              points_name:Tuple[np.ndarray,list,None],
                              output:str):
    from datetime import datetime

    num_time   = rains.shape[1]

    tform = "%Y-%m-%d %H:%M:%S"
    tstamp_now = int(datetime.now().timestamp()/3600/24)*3600*24 - 8*3600
    tstamp_array = np.arange(tstamp_now, tstamp_now+num_time*3600, 3600)
    Tseries = []
    for tstamp in tstamp_array:
        datime = datetime.fromtimestamp(tstamp)
        Tseries.append( datime.strftime(tform) )
    Tseries = np.array(Tseries)

    RD_obj = RainData(rains,points,Tseries)
    if points_name is not None:
        if isinstance(points_name,list): 
            RD_obj.points_name = np.ndarray(points_name,dtype=str)
        else:
            RD_obj.points_name = points_name
    RD_obj.Generate_jsonfile(output)

    return RD_obj

def Calc_AreaRain(GridPolys,basin_poly,input_raindata,coverArea,output=outdir+'Calc_AreaRain.log',identify_limit=1):
    # print("calculating area rain ...")

    #是否输出log文件
    if output == None:
        fw = None
    else:
        fw = open(output,'w')  

    num_grid  = len(GridPolys)

    # 根据是否在流域内的标准，选取raindata
    basin_poly_points = np.array( basin_poly.exterior.coords[:] )
    bool = points_in_poly(input_raindata.points,basin_poly_points) #选取流域内的，有降雨数据的点
    RD_points = input_raindata.points[bool]
    RD_rains  = input_raindata.rains[bool]
    raindata = RainData(RD_rains,RD_points)
    
    if len(RD_points) == 0:
        num_time  = np.size(RD_rains,1)
        WriteLog(fw,"raindata: %d grid points with %d values in terms of time.\n\n"
            %(np.size(RD_rains,0),num_time))
        AreaRain  = np.zeros((num_grid,num_time))
        nearpoint_geo = ops.nearest_points(input_raindata.points_geo,basin_poly)[0] #第一个几何对象的最邻近点
        nearpoint = nearpoint_geo.coords[:][0]
        for i in range(num_grid):
            if coverArea[i] > (1e-3):
                AreaRain[i] = input_raindata.points_dict[nearpoint]
            WriteLog(fw,'cover area: %.2f\n'%coverArea[i])
            WriteLog(fw,'Area Rain in the grid: {0}\n'.format(AreaRain[i]))
            WriteLog(fw,'='*50)
            WriteLog(fw,'\n\n')
        WriteLog(fw,'AreaRain = \n')
        WriteLog(fw,str(AreaRain))
        CloseLog(fw)
        # print("="*100)
        return AreaRain
    else:
        num_time  = np.size(RD_rains,1)
        WriteLog(fw,"raindata: %d grid points with %d values in terms of time.\n\n"
            %(np.size(RD_rains,0),num_time))
        AreaRain  = np.zeros((num_grid,num_time))        

    num_time  = np.size(RD_rains,1)
    WriteLog(fw,"raindata: %d grid points with %d values in terms of time.\n\n"
          %(np.size(RD_rains,0),num_time))
    AreaRain  = np.zeros((num_grid,num_time))
    
    # 计算每个网格的面降雨
    data_points = dict() 
    array_num_datapoints = np.zeros(num_grid)
    for i in range(num_grid):
        poly = GridPolys[i]
        WriteLog(fw,"grid {0}, init area rain: {1}\n".format(i,AreaRain[i]))

        # # 提取有降雨数据的顶点，一般为在流域内的顶点
        # xy = np.array(poly.exterior.coords[:])
        # poly_center = np.array( poly.centroid.coords[:][0] ) 
        # #xy = np.vstack((poly_center,xy)) #添加中心点
        # data_points[poly] = []
        # error_coords = 0
        # for point in xy[0:-1]: #找到每个网格上，有降雨数据的顶点在raindata里的索引
        #     for index,RDpoint in enumerate(RD_points):  
        #         error_coords = np.linalg.norm(point - RDpoint)
        #         if error_coords < identify_limit: #点之间的距离小于identify_limit，即基本相等
        #             data_points[poly].append(index)
        #             WriteLog(fw,'point: %r\tindex in raindata: %d\n'%(point,index))

        bufferpoly = Buffer_grid(poly,1.0)[0]
        data_points[poly] = []
        for index,RDpoint in enumerate(RD_points): 
            if bufferpoly.intersects(Point(RDpoint)):
                data_points[poly].append(index)

        num_datapoints = len(data_points[poly])
        array_num_datapoints[i] = num_datapoints
        WriteLog(fw,"%d points in the grid have rain data.\n"%num_datapoints)
        WriteLog(fw,'cover area: %.2f\n'%coverArea[i])

        # 计算网格面雨量
        if data_points[poly]: #如果网格上的顶点存在降雨数据，则算术平均得到面雨量
            for index in data_points[poly]:
                AreaRain[i] += RD_rains[index]
            AreaRain[i] = AreaRain[i]/num_datapoints
            AreaRain[i] = np.round(AreaRain[i],2)
        else: #如果网格上的顶点不存在降雨数据
            if coverArea[i] > (1e-3): #若分析重叠区面积不为零，则寻找最邻近有降雨数据的顶点作为网格面雨量；否则面雨量为零
                cover_geo = poly.intersection(basin_poly)
                nearpoint_geo = ops.nearest_points(raindata.points_geo,cover_geo)[0] #第一个几何对象的最邻近点
                nearpoint = nearpoint_geo.coords[:][0]
                AreaRain[i] = raindata.points_dict[nearpoint]
                AreaRain[i] = np.round(AreaRain[i],2)
        
        WriteLog(fw,'Area Rain in the grid: {0}\n'.format(AreaRain[i]))
        WriteLog(fw,'='*50)
        WriteLog(fw,'\n\n')

    if sum(array_num_datapoints) == 0:
        # print('Warning: all grids can not find the closest rain data points!')
        pass

    WriteLog(fw,'AreaRain = \n')
    WriteLog(fw,str(AreaRain))
    CloseLog(fw)
    # print("="*100)
    return AreaRain

def Calc_BasinAreaRain(AreaRain,coverArea):
    # print("caculating area rain in the basin ...")
    sumArea = np.sum(coverArea)
    #print("The basin area is %.2f"%sumArea)
    BasinAreaRain = np.dot(coverArea,AreaRain)/sumArea
    BasinAreaRain = np.round(BasinAreaRain,2)
    #print("The area rain in the basin is")
    #print(BasinAreaRain)
    # print("="*100)
    return BasinAreaRain

def AreaRain_anlysis(gridshp,basinshp,raindata,output=None,debug=False):

    if str(type(basinshp)) == "<class 'str'>":
        GridPolys   = ReadGrid(gridshp)[0]
        basin_polys = ReadPolys(basinshp)[0]
        # print(len(basin_polys))
        num_basin   = len(basin_polys)

        if num_basin == 1:
            basin_poly  = basin_polys[0]
            coverArea   = Intersection_analysis(gridshp,basin_poly)
            AreaRain    = Calc_AreaRain(GridPolys,basin_poly,raindata,coverArea,output=output)
            BasinAreaRain = Calc_BasinAreaRain(AreaRain,coverArea)
            if debug: 
                new_field = {'arearain':AreaRain[:,0]}
                Add_field(gridshp,new_field)
            return coverArea,AreaRain,BasinAreaRain
        else:
            coverArea_list = []
            AreaRain_list  = []
            BasinAreaRain_list = np.zeros( (num_basin,raindata.num_time) )
            for i,basin_poly in enumerate(basin_polys):
                # print(i)
                coverArea = Intersection_analysis(gridshp,basin_poly)
                AreaRain  = Calc_AreaRain(GridPolys,basin_poly,raindata,coverArea,
                                          output=output)
                coverArea_list.append( coverArea )
                AreaRain_list.append( AreaRain )
                BasinAreaRain_list[i] = Calc_BasinAreaRain(AreaRain,coverArea)
            if debug: 
                new_field = {'arearain':AreaRain_list[10][:,0]} #最后一个流域的，第一个时刻的网格面雨量
                Add_field(gridshp,new_field)
            return coverArea_list,AreaRain_list,BasinAreaRain_list

    elif str(type(basinshp)) == "<class 'shapely.geometry.polygon.Polygon'>":
        GridPolys  = ReadGrid(gridshp)[0]
        basin_poly = basinshp
        coverArea  = Intersection_analysis(gridshp,basin_poly)
        AreaRain   = Calc_AreaRain(GridPolys,basin_poly,raindata,coverArea)
        BasinAreaRain = Calc_BasinAreaRain(AreaRain,coverArea) 
        if debug: 
            new_field = {'arearain':AreaRain[:,0]}
            Add_field(gridshp,new_field)
        return coverArea,AreaRain,BasinAreaRain   
    else:
        raise TypeError("Wrong type of basinshp in AreaRain_anlysis!")

def Get_extent(poly):
    bd = poly.bounds
    bd_points = [
        [bd[0],bd[1]],
        [bd[2],bd[1]],
        [bd[2],bd[3]],
        [bd[0],bd[3]],
        [bd[0],bd[1]]
    ]
    # print(bd_points)
    return Polygon(bd_points)

def Update_GridPolys(basin_poly,GridPolys):
    extent = Get_extent(basin_poly)
    min_Basin_GridPolys = []
    for poly in GridPolys:
        if poly.intersects(extent):
            min_Basin_GridPolys.append(poly)
    return

def Transform_CRS(points,EPSG_init,EPSG_target,output=None):
    from pyproj import CRS,Transformer

    in_proj  = CRS.from_epsg(EPSG_init)
    out_proj = CRS.from_epsg(EPSG_target)
    # print(" ~> Projecting from {} to {}".format(in_proj.name,
    #                                            out_proj.name))
    transformer = Transformer.from_crs(in_proj, out_proj, always_xy=True)
    newx, newy = transformer.transform(points[:,0],points[:,1])

    data = np.column_stack((points,newx,newy))
    if output != None:
        if (EPSG_target == 4326) or (EPSG_init == 4326):
            np.savetxt(output,data,delimiter=',',fmt='%.6f',encoding="utf-8",
                    header='x,y,newx,newy')
        else:
            np.savetxt(output,data,delimiter=',',fmt='%.2f',encoding="utf-8",
                    header='x,y,newx,newy')
    return np.array([newx,newy]).T


#测站点数据raindata
#每个插值点的最邻近测站——插值点到各测站的距离
#插值点

def Calc_distance(points,origin):

    if str(type(points)) != "<class 'numpy.ndarray'>":
        points = np.array(points)
    if str(type(origin)) != "<class 'numpy.ndarray'>":
        origin = np.array(origin)

    if origin.shape == (2,):
        distance_array = np.array(len(points))
        distance_array = np.sqrt(np.sum((points - origin) ** 2, 1))
        return distance_array/1000
    elif len(origin) > 1 or origin.shape == (1,2):
        Dist = np.zeros((len(origin),len(points)))
        for iter,item in enumerate(origin):
            Dist[iter,:] = np.sqrt(np.sum((points - item) ** 2, 1))
        return Dist/1000

class Nearest_Analyzer():
    def __init__(self,points,interpolated_points,points_identity=None):
        self.isinit = 'YES' #判别是否为初始化的变量
        
        self.points  = points
        self.interpolated_points = interpolated_points
        self.num_points    = len(points)
        self.num_inpoints  = len(interpolated_points)
        self.Dist          = Calc_distance(points,interpolated_points)
        self.Index_sorted, self.Order, self.Dist_sorted, self.Points_sorted_dict = self.Sort_Distance()
        #self.bool_selected   = None
        #self.points_selected = None
        self.points_identity_sorted = None
        self.points_identity = points_identity

        self.isinit = 'No' #初始化结束

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if self.isinit == 'No':
            if key == 'points':
                self.__dict__['num_points'] = len(value)
                self.__dict__['Dist'] = Calc_distance(value,self.interpolated_points)
                Index_sorted,Order,Dist_sorted,Points_sorted_dict = self.Sort_Distance()
                self.__dict__['Index_sorted'] = Index_sorted
                self.__dict__['Order']        = Order  
                self.__dict__['Dist_sorted']  = Dist_sorted
                self.__dict__['Points_sorted_dict'] = Points_sorted_dict
            elif key == 'interpolated_points':
                self.__dict__['num_inpoints'] = len(value)
                self.__dict__['Dist'] = Calc_distance(self.points,value)
                Index_sorted,Order,Dist_sorted,Points_sorted_dict = self.Sort_Distance()
                self.__dict__['Index_sorted'] = Index_sorted
                self.__dict__['Order']        = Order  
                self.__dict__['Dist_sorted']  = Dist_sorted
                self.__dict__['Points_sorted_dict'] = Points_sorted_dict
            elif key == 'points_identity' and list(value) != None: #array用!=None判断会报错
                if "array" not in str(type(value)):
                    self.__dict__[key] = np.array(value)
                Identity_sorted = np.array([value]*self.num_inpoints)
                for i,distance_sorted_index in enumerate(self.Index_sorted):
                    Identity_sorted[i] = self.points_identity[distance_sorted_index] 
                self.__dict__['points_identity_sorted'] = Identity_sorted
        elif self.isinit == 'YES':
            # print("intializing 'Interpolation_Points' Object...")
            pass

                        
    def Sort_Distance(self):
        Dist_sorted   = np.zeros((self.num_inpoints,self.num_points))
        Index_sorted  = np.zeros((self.num_inpoints,self.num_points),dtype=int)
        Order         = np.zeros((self.num_inpoints,self.num_points),dtype=int)
        
        Points_sorted_dict = dict()
        for i,distance_array in enumerate(self.Dist):
            #example
            #distance_array: [5 0 3]
            #index_sorted:  [1 2 0]
            #order:        [2 0 1]
            #distance_array[index_sorted]: [0 3 5]
            index_sorted       = distance_array.argsort()
            Dist_sorted[i]     = distance_array[index_sorted]
            Index_sorted[i]    = index_sorted
            for j in range(self.num_points):
                Order[i][index_sorted[j]] = j #distance_array各元素的大小顺序
            
            inpoint = tuple(self.interpolated_points[i])
            Points_sorted_dict[i] = self.points[index_sorted]
        return Index_sorted,Order,Dist_sorted,Points_sorted_dict

    # 寻找某个插值点的最邻近N个参证点
    def Select_by_N(self,N,inpoint_index,mode='normal'): 
        if inpoint_index < self.num_inpoints:
            if mode == 'for sorted array':
                bool = np.arange(0,self.num_points,1,dtype=int) < N
                return bool
            elif mode == 'normal':
                bool = self.Order[inpoint_index] < N
                return bool
            else:
                raise ValueError("not an implemented mode.")
        else:
            raise IndexError("The index of interpolated points is out of range!")
    
    # 寻找与某个插值点距离小于R的参证点
    def Select_by_R(self,R,inpoint_index,mode='normal'):
        if inpoint_index < self.num_inpoints:
            if mode == 'for sorted array':
                bool = self.Dist_sorted[inpoint_index] <= R
                return bool
            elif mode == 'normal':
                bool = self.Dist[inpoint_index] <= R
                return bool
            else:
                raise ValueError("not an implemented mode.")
        else:
            raise IndexError("The index of interpolated points is out of range!")

    def IDW_mod(self,basinshp,raindata,
                dmin,Rint,Rmax,N,p=2,method=2,
                output=outdir+'IDW_mod.log'):
        
        if output == None:
            fw = None
        else:
            fw = open(output,'w',encoding="utf-8")

        IDW_data = np.zeros((self.num_inpoints,raindata.num_time))
        Ref_stations = [[]]*self.num_inpoints
        
        basin_poly = ReadBasin(basinshp)[0]
        basin_poly_points = np.array( basin_poly.exterior.coords[:] )
        bool_basin_inpoints = points_in_poly(self.interpolated_points,basin_poly_points) #选取流域内的插值点
        bool_basin_points   = points_in_poly(self.points,basin_poly_points) #选取流域内的参证点
        
        # 将self中的points更改为流域内的点，再更改对应的point_identity
        self.points = self.points[bool_basin_points]
        self.points_identity = self.points_identity[bool_basin_points]
        # print("The points were changed to be those in the basin!")

        # 流域外的网格顶点
        for inpoint_index in np.arange(self.num_inpoints)[~bool_basin_inpoints]:
            IDW_data[inpoint_index] = np.ones((1,raindata.num_time)) * np.nan
        # 流域内的网格顶点
        for inpoint_index in np.arange(self.num_inpoints)[bool_basin_inpoints]:
            # 初始化
            dist = np.array([])
            values = np.array([])

            WriteLog(fw,'='*80+"\n")
            WriteLog(fw,'%d\n'%(inpoint_index+1))

            WriteLog(fw,'1st-(N+1)th P:\t{0}\n'.format(self.points_identity_sorted[inpoint_index][:N+1]))
            WriteLog(fw,'1st-(N+1)th D:\t{0}\n\n'.format(self.Dist_sorted[inpoint_index][:N+1]))

            # 选取距离小于dmin的最邻近点作为参证点
            bool = self.Select_by_R(dmin,inpoint_index,mode='for sorted array')
            station = self.points_identity_sorted[inpoint_index][bool]
            WriteLog(fw,'dmin={0}\t{1}\n'.format(dmin,station))
            if len(station) != 0:
                Ref_stations[inpoint_index] = station[0:1]
                values = raindata.Get_data_by_names(station[0:1])
                dist = self.Dist_sorted[inpoint_index][bool][0:1]
                IDW_data[inpoint_index] = self.IDW_mod_interpolation(dist,values,m=1)
                WriteLog(fw,'distance\t{0}\n'.format(dist))
                WriteLog(fw,'IDW rain\t{0}\n'.format(IDW_data[inpoint_index]))
                continue
            
            # 不存在距离小于dmin的最邻近点，根据N/R寻找参证点
            bool = self.Select_by_R(Rint,inpoint_index,mode='for sorted array')
            station = self.points_identity_sorted[inpoint_index][bool]
            if len(station) >= N:
                bool = self.Select_by_N(N,inpoint_index,mode='for sorted array')
                station = self.points_identity_sorted[inpoint_index][bool]
                WriteLog(fw,'Rint={0}\t{1}\n'.format(Rint,station))
                Ref_stations[inpoint_index] = station
                values = raindata.Get_data_by_names(station)
                dist = self.Dist_sorted[inpoint_index][bool]
                IDW_data[inpoint_index] = self.IDW_mod_interpolation(dist,values,m=2)                
                WriteLog(fw,'distance\t{0}\n'.format(dist))
                WriteLog(fw,'IDW rain\t{0}\n'.format(IDW_data[inpoint_index]))
            else:
                WriteLog(fw,'Rint={0}\t{1}\n'.format(Rint,station))
                bool_Rmax = self.Select_by_R(Rmax,inpoint_index,mode='for sorted array')
                station_Rmax = self.points_identity_sorted[inpoint_index][bool_Rmax]
                if len(station_Rmax) >= N:
                    bool = self.Select_by_N(N,inpoint_index,mode='for sorted array')
                    station = self.points_identity_sorted[inpoint_index][bool]
                    WriteLog(fw,'Rmax={0}\t{1}\n'.format(Rmax,station))
                    values = raindata.Get_data_by_names(station)
                    dist = self.Dist_sorted[inpoint_index][bool]
                    IDW_data[inpoint_index] = self.IDW_mod_interpolation(dist,values,m=2)
                else:
                    bool = bool_Rmax
                    station = station_Rmax
                    if len(station) == 0:
                        # print("Warning: Can not find ref. points for interpolated point %d"%(inpoint_index+1))
                        WriteLog(fw,'Rmax={0}\t{1}\n'.format(Rmax,station))
                        WriteLog(fw,"Warning: Can not find ref. points for interpolated point\n")
                        IDW_data[inpoint_index] = np.ones((1,raindata.num_time)) * 0.
                    else:
                        WriteLog(fw,'Rmax={0}\t{1}\n'.format(Rmax,station))
                        dist = self.Dist_sorted[inpoint_index][bool]
                        values = raindata.Get_data_by_names(station)
                        IDW_data[inpoint_index] = self.IDW_mod_interpolation(dist,values,m=method)   
                                 
                Ref_stations[inpoint_index] = station
                WriteLog(fw,'distance\t{0}\n'.format(dist))
                WriteLog(fw,'IDW rain\t{0}\n'.format(IDW_data[inpoint_index]))
        CloseLog(fw)

        return IDW_data,Ref_stations

    def IDW_mod_region_dependent(self,
                basinshp,raindata,
                para,regionshp,regionname,
                output=outdir+'IDW_mod.log'):
        # print("Modified inverse distance interpolation (region-dependent) ...")

        if output == None:
            fw = None
        else:
            fw = open(output,'w',encoding="utf-8")
        IDW_data = np.zeros((self.num_inpoints,raindata.num_time))
        Ref_stations = [[]]*self.num_inpoints
        
        basin_poly = ReadBasin(basinshp)[0]
        basin_poly_points = np.array( basin_poly.exterior.coords[:] )
        bool_basin_inpoints = points_in_poly(self.interpolated_points,basin_poly_points) #选取流域内的插值点
        bool_basin_points   = points_in_poly(self.points,basin_poly_points) #选取流域内的参证点
        
        # 将self中的points更改为流域内的点，再更改对应的point_identity
        self.points = self.points[bool_basin_points]
        self.points_identity = self.points_identity[bool_basin_points]
        # print("The points were changed to be those in the basin!")

        # 流域外的网格顶点
        for inpoint_index in np.arange(self.num_inpoints)[~bool_basin_inpoints]:
            IDW_data[inpoint_index] = np.ones((1,raindata.num_time)) * np.nan

        # 流域内的网格顶点
        regPolygons = ReadPolys(regionshp)[0]
        regnames    = ReadTable(regionshp,regionname)
        region_id = np.zeros(self.num_inpoints)
        for ireg,region in enumerate(regPolygons):

            region_poly_points   = np.array( region.exterior.coords[:] )
            bool_region_inpoints = points_in_poly(self.interpolated_points,region_poly_points)

            para_region = para[ regnames[ireg] ]
            dmin,Rint,Rmax,N,p,method = para_region
            # print(regnames[ireg],dmin,Rint,Rmax,N,p,method)

            # 检查对插值点的分类是否正确
            bool_basin_region = bool_basin_inpoints & bool_region_inpoints
            region_id[bool_basin_region] = np.ones(len(region_id[bool_basin_region])) * (ireg+1)

            for inpoint_index in np.arange(self.num_inpoints)[bool_basin_region]:
                # 初始化
                dist = np.array([])
                values = np.array([])

                WriteLog(fw,'='*80+"\n")
                WriteLog(fw,'%d\n'%(inpoint_index+1))

                WriteLog(fw,'1st-(N+1)th P:\t{0}\n'.format(self.points_identity_sorted[inpoint_index][:N+1]))
                WriteLog(fw,'1st-(N+1)th D:\t{0}\n\n'.format(self.Dist_sorted[inpoint_index][:N+1]))

                # 选取距离小于dmin的最邻近点作为参证点
                bool = self.Select_by_R(dmin,inpoint_index,mode='for sorted array')
                station = self.points_identity_sorted[inpoint_index][bool]
                WriteLog(fw,'dmin={0}\t{1}\n'.format(dmin,station))
                if len(station) != 0:
                    Ref_stations[inpoint_index] = station[0:1]
                    values = raindata.Get_data_by_names(station[0:1])
                    dist = self.Dist_sorted[inpoint_index][bool][0:1]
                    IDW_data[inpoint_index] = self.IDW_mod_interpolation(dist,values,m=1)
                    WriteLog(fw,'distance\t{0}\n'.format(dist))
                    WriteLog(fw,'IDW rain\t{0}\n'.format(IDW_data[inpoint_index]))
                    continue
                
                # 不存在距离小于dmin的最邻近点，根据N/R寻找参证点
                bool = self.Select_by_R(Rint,inpoint_index,mode='for sorted array')
                station = self.points_identity_sorted[inpoint_index][bool]
                if len(station) >= N:
                    bool = self.Select_by_N(N,inpoint_index,mode='for sorted array')
                    station = self.points_identity_sorted[inpoint_index][bool]
                    WriteLog(fw,'Rint={0}\t{1}\n'.format(Rint,station))
                    Ref_stations[inpoint_index] = station
                    values = raindata.Get_data_by_names(station)
                    dist = self.Dist_sorted[inpoint_index][bool]
                    IDW_data[inpoint_index] = self.IDW_mod_interpolation(dist,values,m=2)                
                    WriteLog(fw,'distance\t{0}\n'.format(dist))
                    WriteLog(fw,'IDW rain\t{0}\n'.format(IDW_data[inpoint_index]))
                else:
                    WriteLog(fw,'Rint={0}\t{1}\n'.format(Rint,station))
                    bool_Rmax = self.Select_by_R(Rmax,inpoint_index,mode='for sorted array')
                    station_Rmax = self.points_identity_sorted[inpoint_index][bool_Rmax]
                    if len(station_Rmax) >= N:
                        bool = self.Select_by_N(N,inpoint_index,mode='for sorted array')
                        station = self.points_identity_sorted[inpoint_index][bool]
                        WriteLog(fw,'Rmax={0}\t{1}\n'.format(Rmax,station))
                        values = raindata.Get_data_by_names(station)
                        dist = self.Dist_sorted[inpoint_index][bool]
                        IDW_data[inpoint_index] = self.IDW_mod_interpolation(dist,values,m=2)
                    else:
                        bool = bool_Rmax
                        station = station_Rmax
                        if len(station) == 0:
                            # print("Warning: Can not find ref. points for interpolated point %d"%(inpoint_index+1))
                            WriteLog(fw,'Rmax={0}\t{1}\n'.format(Rmax,station))
                            WriteLog(fw,"Warning: Can not find ref. points for interpolated point\n")
                            IDW_data[inpoint_index] = np.ones((1,raindata.num_time)) * 0.
                        else:
                            WriteLog(fw,'Rmax={0}\t{1}\n'.format(Rmax,station))
                            dist = self.Dist_sorted[inpoint_index][bool]
                            values = raindata.Get_data_by_names(station)
                            IDW_data[inpoint_index] = self.IDW_mod_interpolation(dist,values,m=method)   
                    
                    Ref_stations[inpoint_index] = station
                    WriteLog(fw,'distance\t{0}\n'.format(dist))
                    WriteLog(fw,'IDW rain\t{0}\n'.format(IDW_data[inpoint_index]))
        CloseLog(fw)

        # region_test = np.column_stack( (self.interpolated_points,region_id) )
        # np.savetxt(outdir+"region_test.txt",region_test,delimiter=',',fmt="%.2f",encoding="utf-8")

        return IDW_data,Ref_stations

    def IDW_mod_interpolation(self,distances,values,p=2,m=2):
        if m == 2:
            weights = 1 / (distances ** p)
            weight_sum = np.sum(weights)
            interpolated_value = np.dot(weights,values) / weight_sum
        if m == 1:
            interpolated_value = values[0]
        if m == 3:
            weights = 1 / (distances ** p)
            weight_sum = np.sum(weights)
            interpolated_value = np.dot(weights,values)       
        return np.round(interpolated_value,2)

def Buffer_grid(grid:Polygon,buf:float):
    points = np.array( grid.exterior.coords[:] )
    center = np.array( grid.centroid.coords[:][0] )
    new_points = []
    for point in points:
        vector = point - center
        if (vector[0] > 0) and (vector[1] > 0):
            new_points.append( point + buf )
        elif (vector[0] < 0) and (vector[1] < 0):
            new_points.append( point - buf )
        elif (vector[0] > 0) and (vector[1] < 0):
            new_points.append( point + np.array([buf,-buf]) )
        else:
            new_points.append( point + np.array([-buf,buf]) )
    new_points = np.array(new_points)
    new_grid = Polygon(new_points)
    return new_grid,new_points

def Change_list_order(list,order):
    new_list = []
    for iter in order:
        new_list.append(list[iter])
    return new_list

def Calc_Voronoi_ratio(Voronoi_shp,name_voro,
                       subbasinshp,name_subbasin,
                       sort_voro=None,sort_subbasin=None,
                       output=None):
    
    Voronois  = ReadPolys(Voronoi_shp)[0]
    Subbasins = ReadPolys(subbasinshp)[0]

    if sort_voro != None:
        Voronoi_names,order_voro = ReadTable(Voronoi_shp,name_voro,sort=sort_voro)
        Voronois = Change_list_order(Voronois,order_voro)
    else:
        Voronoi_names = ReadTable(Voronoi_shp,name_voro)
    if sort_subbasin != None:
        subbasin_names,order_subbasin = ReadTable(subbasinshp,name_subbasin,sort=sort_subbasin)
        Subbasins = Change_list_order(Subbasins,order_subbasin)
    else:
        subbasin_names = ReadTable(subbasinshp,name_subbasin)
    
    num_Voronoi  = len(Voronoi_names)
    num_subbasin = len(subbasin_names)

    ratio = np.zeros((num_subbasin,num_Voronoi))
    for i,subbasin in enumerate(Subbasins):
        subbasin_area = subbasin.area
        for j,voronoi in enumerate(Voronois):
            if subbasin.intersects(voronoi):
                common_part = subbasin.intersection(voronoi)
                ratio[i,j] = common_part.area / subbasin_area
            else:
                ratio[i,j] = 0.0
            

    if output != None:
        Header = "\t".join(item for item in (['子流域'] + Voronoi_names))
        with open(output,'w',encoding="utf-8") as fw:
            fw.write(Header+"\n")
            for i in range(num_subbasin):
                line = "%s" + "\t%.3f"*num_Voronoi + "\n"
                fw.write(line%tuple(subbasin_names[i:i+1] + list(ratio[i])))
    
    # print("Finished")
    return ratio


def Calc_PRCP_IDW(  parafile:str,RData:RainData,
                    gridshp:str,basinshp:str,choiceshp:str,
                    voronoi_shp:str,voro_name_field:str,
                    sub_basinshp:str,subB_name_field:str,subB_sort_field:str,
                    debug:bool ):
    #读取网格化降雨参数 - 文件中站点顺序不重要，参数都为整数型
    meshpara = np.loadtxt(parafile,delimiter=',', skiprows=1, encoding="utf-8", dtype=str)
    para_voro = dict()
    for i in range(meshpara.shape[0]):
        para_voro[meshpara[i,0]] = np.int32(np.array(meshpara[i,1:],dtype=float))

    #最邻近分析
    interpolated_points = Get_Grid_points(gridshp,output=None,
                                            isdensify=None)
    NA = Nearest_Analyzer(RData.points,interpolated_points)

    #网格化插值
    NA.points_identity = RData.points_name
    IDW_data = NA.IDW_mod_region_dependent(choiceshp,RData,
                para_voro,voronoi_shp,voro_name_field,
                output=None)[0]

    bool = np.ones(NA.num_inpoints) > 0
    for i,item in enumerate(IDW_data):
        if True in np.isnan(item):
            bool[i] = False
    mesh_raindata = RainData(IDW_data[bool],interpolated_points[bool])

    #网格化子流域面雨量
    arearain = AreaRain_anlysis(gridshp,basinshp,mesh_raindata,debug=debug)[1] #统一的网格面雨量
    BasinPolys = ReadPolys(sub_basinshp)[0]
    basinarearain = np.zeros( (len(BasinPolys),mesh_raindata.num_time) )
    for iter,basin_poly in enumerate(BasinPolys):
        coverarea = Intersection_analysis(gridshp,basin_poly,debug=debug)
        basinarearain[iter] = Calc_BasinAreaRain(arearain,coverarea)

    names_basin_sorted,order = ReadTable(sub_basinshp,subB_name_field,subB_sort_field)
    basinarearain = basinarearain[order] #对网格化子流域面雨量进行排序
    basinarearain = np.round(basinarearain,2)
    if debug:
        new_Field = {'arearain':basinarearain[:,0]}
        Add_field(sub_basinshp,new_Field)
    return basinarearain