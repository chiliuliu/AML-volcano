from obspy import read
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
import datetime
import os
from datetime import datetime, timedelta
import numpy as np
from statistics import mean
n_jobs=-1

def fill_data(front,sample_rate):
    re_rate=sample_rate
    sa_rate=100
    st=read(front)
    st.filter("bandpass", freqmin=.5, freqmax=30)
    mean_data=mean(st[0].data)
    st.resample(re_rate)
    data=st[0].data
    fill = np.full(int(8640000/sa_rate*re_rate)-len(data),mean_data)
    start_time=st[0].meta.starttime.strftime("%H %M %S")
    if(start_time=="00 00 00"):
        data=np.insert(data,data.shape[0],fill)
    else:
        data=np.insert(data,0,fill)
    return data


def get_windows_feature(front,year,sensor,gap_hour,size_hour,start,end,sample_rate,data_save,overwrite=False):
    print(str(year)+" "+str(sensor)+" "+str(gap_hour)+" "+str(size_hour))
    #for recording all data
    df_data=pd.DataFrame()
    #predefine of the constant
    windows=pd.DataFrame()
    #total_num_a_day need to be modified before
    total_num_a_day=8640000/100*sample_rate
    gap_window=total_num_a_day/(24/gap_hour)
    size_window=total_num_a_day/(24/size_hour)
    start_time=datetime.now()
    end_time=datetime.now()
    files=["HHE.D","HHN.D","HHZ.D"]
    df_save=data_save+sensor+"_"+str(year)+"_"+str(sample_rate)+"_"+str(start)+"_"+str(end)+".csv"
    if(os.path.exists(df_save)):
        df_data=pd.read_csv(df_save,index_col=0)
    else:
        #get data from the file and concat them together to df_data
        for i in range(start,end+1):
            if (i-start)%30==0:
                print("Data extraction:"+str(i-start)+" out of "+str(end-start+1))
            df_three_file_data=pd.DataFrame() 
            for file in files:
                front_temp=front+file+"/UC."+sensor+".00."+file+"."+str(year)+"."
                if i<10:
                    filename=front_temp+"00"+str(i)
                elif i<100:
                    filename=front_temp+"0"+str(i)
                else:
                    filename=front_temp+str(i)

                st=read(filename)
                trace=st[0]
                st.filter("bandpass", freqmin=.5, freqmax=30)
                tr=trace.resample(sample_rate)#!!!!!!!!!!!!
                temp_df=pd.DataFrame(tr.data)
                if(len(temp_df)==total_num_a_day):
                    if(len(df_three_file_data)==0):
                        df_three_file_data=pd.DataFrame(tr.data,columns=[file])
                    else:
                        df_three_file_data.insert(0,file,temp_df)
                else:#the missing data
                    temp_df=fill_data(filename,sample_rate)
                    if(len(df_three_file_data)==0):
                        df_three_file_data=pd.DataFrame(temp_df,columns=[file])
                    else:
                        df_three_file_data.insert(0,file,temp_df)


            df_data=pd.concat([df_data,df_three_file_data])

            #ignore when overwritting=True
            if i==start:
                start_time=tr.meta.starttime
                #if start_time,gap_hour,size_hour exists, break;
                if(os.path.exists("./"+sensor+"_data/"+sensor+","+start_time.strftime("%Y-%m-%d %H:%M:%S")+","+(start_time+timedelta(hours=size_hour)).strftime("%Y-%m-%d %H:%M:%S")+".csv")):
                    if(os.path.exists("./"+sensor+"_data/"+sensor+","+(start_time+timedelta(hours=gap_hour)).strftime("%Y-%m-%d %H:%M:%S")+","+(start_time+timedelta(hours=size_hour+gap_hour)).strftime("%Y-%m-%d %H:%M:%S")+".csv")):
                        if(not overwrite):
                            print("file exists")
                            return
        #can save data here,sensor_year_sampling_rate_start_end
        df_data.to_csv(df_save)


    #get sliding windows
    begin=0
    end=begin+size_window
    begin=0
    end=begin+size_window
    move_times=int((len(df_data)-size_window)/gap_window)
    for i in range(move_times) : #total movement=(length-size)/gap
        if i%(30*12)==0:
            print("Sliding windows:"+str(i)+" out of "+str(move_times))
        temp_window=pd.DataFrame(df_data.iloc[int(begin):int(end),[0,1,2]])
        #data=temp_window.values

        temp_window.insert(0,"id",i)
        #temp_window=pd.DataFrame(data.reshape(-1,1),columns=['feature'])
        #temp_window['id']=i  ##

        begin=begin+gap_window
        end=end+gap_window
        windows=pd.concat([windows,temp_window])
    X = extract_features(windows, column_id='id', n_jobs=16, default_fc_parameters=EfficientFCParameters()) 
    temp=pd.DataFrame(X)
    return temp

def year_data(year,days,front,sensor,gap,size,sample_rate,savepath,data_save):
    result=pd.DataFrame()
    for day in days:
        temp=get_windows_feature(front,year,sensor,gap,size,day[0],day[1],sample_rate,data_save,overwrite=True)
        result=pd.concat([result,temp])
    recordname=savepath+str(gap)+"_"+str(size)+"_"+str(sample_rate)+"_"+str(year)+"_"+sensor+".csv"
    result.to_csv(recordname)



if __name__ == '__main__':
    gap_list=[1,1,2,4,4,4]
    size_list=[2,4,6,8,10,12]
    # gap_list=[4]
    # size_list=[12]
    sample_rate=0.01

    sensor="WEST"
    data_save="/home/s3126161/data/hf_data_save/"
    savepath="/home/s3126161/data/hf_result_save/"
    for i in range(len(gap_list)):
        gap=gap_list[i]
        size=size_list[i]
        front="/data/colima-share/database/SDS/2013/UC/WEST/"
        year_data(2013,[[141,365]],front,sensor,gap,size,sample_rate,savepath,data_save)
        front="/data/colima-share/database/SDS/2014/UC/WEST/"
        year_data(2014,[[1,365]],front,sensor,gap,size,sample_rate,savepath,data_save)
        front="/data/colima-share/database/SDS/2015/UC/WEST/"
        year_data(2015,[[1,171],[182,283]],front,sensor,gap,size,sample_rate,savepath,data_save)
        front="/data/colima-share/database/SDS/2016/UC/WEST/"
        year_data(2016,[[19,366]],front,sensor,gap,size,sample_rate,savepath,data_save)
        front="/data/colima-share/database/SDS/2017/UC/WEST/"
        year_data(2017,[[1,15],[41,151]],front,sensor,gap,size,sample_rate,savepath,data_save)

    # savepath="./"
    # front="./WEST/"
    # data_save="./data_save/"
    # year_data(2014,[[1,15],[41,151]],front,sensor,gap,size,sample_rate,savepath,data_save)


    # year=2015
    # days=[[1,171],[182,283]]
    # result=pd.DataFrame()
    # for day in days:
    #     temp=get_windows_feature(front,year,sensor,gap,size,day[0],day[1],sample_rate,overwrite=True)
    #     result=pd.concat([result,temp])
    # recordname=savepath+str(gap)+"_"+str(size)+"_"+str(sample_rate)+"_"+str(year)+"_"+sensor+".csv"
    # result.to_csv(recordname)

    # year=2016
    # days=[[19,366]]
    # result=pd.DataFrame()
    # for day in days:
    #     temp=get_windows_feature(front,year,sensor,gap,size,day[0],day[1],sample_rate,overwrite=True)
    #     result=pd.concat([result,temp])
    # recordname=savepath+str(gap)+"_"+str(size)+"_"+str(sample_rate)+"_"+str(year)+"_"+sensor+".csv"
    # result.to_csv(recordname)

    # year=2017
    # days=[[1,15],[41,151]]
    # result=pd.DataFrame()
    # for day in days:
    #     temp=get_windows_feature(front,year,sensor,gap,size,day[0],day[1],sample_rate,overwrite=True)
    #     result=pd.concat([result,temp])
    # recordname=savepath+str(gap)+"_"+str(size)+"_"+str(sample_rate)+"_"+str(year)+"_"+sensor+".csv"
    # result.to_csv(recordname)


    # days=[[1,365]]
    # year=2014
    # result=pd.DataFrame()
    # for day in days:
    #     temp=get_windows_feature(front,year,sensor,gap,size,day[0],day[1],sample_rate,overwrite=True)
    #     result=pd.concat([result,temp])
    # recordname=savepath+str(gap)+"_"+str(size)+"_"+str(sample_rate)+"_"+str(year)+"_"+sensor+".csv"
    # result.to_csv(recordname)

    # days=[[141,365]]
    # year=2013
    # result=pd.DataFrame()
    # for day in days:
    #     temp=get_windows_feature(front,year,sensor,gap,size,day[0],day[1],sample_rate,overwrite=True)
    #     result=pd.concat([result,temp])
    # recordname=savepath+str(gap)+"_"+str(size)+"_"+str(sample_rate)+"_"+str(year)+"_"+sensor+".csv"
    # result.to_csv(recordname)
