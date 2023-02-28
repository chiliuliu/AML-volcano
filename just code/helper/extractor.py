import pickle as pkl

from tsfresh import extract_features

with open('./to_be_extracted.pkl', 'rb') as f:
   data =  pkl.load(f)

#Distributor = LocalDaskDistributor(n_workers=16)

ok = {}
not_ok = {}

proxy = {}
for x in data.combined_id.unique():
    proxy[x] = len(data[data.combined_id==x]) 

proxy_ord = {k: v for k, v in sorted(proxy.items(), key=lambda item: item[1])}

for x in proxy_ord.keys():
   
    data_temp = data[data.combined_id==x]
    
    try:
        print(f'Ok')
        print(f'id: {x} and len: {len(data_temp)}')
        print('\n')
        ok[x] = len(data_temp)
        extracted_features_tmp = extract_features(data_temp, column_id="combined_id",column_sort="time", n_jobs = 32, chunksize = 10, disable_progressbar=True, distributor=None)
        print('\n')
       # extracted_features_tmp.to_csv('2020-03-30-features-extracted_EMG_traces_connected_36.csv')
    except:
        print(f'Not Ok')
        print(f'id: {x} and len: {len(data_temp)}')
        not_ok[x] = len(data_temp)
        print('\n')







