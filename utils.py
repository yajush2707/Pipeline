import pandas as pd
import json
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO
#import spacy
from gensim.models.word2vec import Word2Vec
import gensim
import json
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
import json
from base64 import b64encode
import time
from datetime import datetime
import random
import string

class LogisticWord2VecModel:
    def __init__(self,model_data=None):
        if model_data is None:     
            
            self.model = gensim.models.KeyedVectors.load_word2vec_format("german.model", binary=True)
            self.model.init_sims(replace=True)
        else:
            self.model,self.models=pickle.load(open(model_data,"rb"))

        
    def train(self,train_data):       
        ### import model..
        res=train_data

        self.models={k:self.__train_classifier(res[k]) for k in res}
        return self.models
            
    def __train_classifier(self,data):
        trues=data["trues"]
        falses=data["falses"]

        true_vecs=np.array([self.model.wv[word] for word in trues if (word in self.model.wv)])
        false_vecs=np.array([self.model.wv[word] for word in falses if (word in self.model.wv)])
        train=np.append(true_vecs,false_vecs,axis=0)
        self.clf=LogisticRegression()
        self.clf.fit(train,[1]*len(true_vecs)+[0]*len(false_vecs))
        return self.clf
    
    def predict(self,transaction_data):
        self.flags={}
        self.words={}
        
        # get word vectors
        for sentence in transaction_data:
            tokens=str(sentence).split(" ")
            for word in tokens:
                word = word.title()
                if word in self.model:
                    self.words[word]=self.model[word]
        
        # classify word vectors
        
        for category in self.models:
            self.flags[category]={}
            self.preds=self.models[category].predict(np.array(list(self.words.values())))
            for i,p in enumerate(self.preds):
                self.flags[category][list(self.words.keys())[i]]=p
                            
        # Categorize transactions
        self.flag_list=[]
        for c in transaction_data:
            flag={}
            for category in self.models: 
                flag[category]=self.categorize_sentence(category,str(c))
            self.flag_list+=[flag]
        labels=pd.DataFrame(self.flag_list).fillna(0.)
        return labels
        
    def categorize_sentence(self,category,sentence):
        flags = {}
        tokens=str(sentence).split(" ")#self.nlp(sentence)
        clf_model=self.models[category]
        for word in tokens:
            word = word.title()
            if word in self.words:
                if self.flags[category][word]==1:
                    return 1
        return 0
    
    def persist(self,file_name):
        pickle.dump([self.model,self.models],open(file_name,"wb"))



    
    
def map_DB(data):
    try:
        dat=json.load(data.input)
        df=pd.DataFrame(dat["transactions"][3])

        mapped_df=pd.DataFrame()
        mapped_df["Beguenstigter / Auftraggeber"]=df["counterPartyName"]
        mapped_df["BIC / BLZ"]=df["creditorId"]
        mapped_df["Betrag"]=df["amount"]
        mapped_df["Buchungstag"]=pd.to_datetime(df["bookingDate"])
        mapped_df["Verwendungszweck"]=df["paymentReference"]
        mapped_df["Umsatzart"]="Kartenzahlung/-abrechnung"

        print("Mapping successful: No error found..")
        data.enricher_input=mapped_df.to_csv()
        data.mapping_status="OK"
        return True
    except Exception as e:
        print("this was no DB file\n"+str(e))
        return False

def map_DKB(data):
    try:
        #df=pd.read_csv(data.input,sep=";",skiprows=6,encoding="ISO-8859-1",decimal=b',')
        df=pd.read_csv(data.input,sep=";",encoding="ISO-8859-1",decimal=b',')
        mapped_df=pd.DataFrame()
        mapped_df["BIC / BLZ"]=df["BLZ"]
        mapped_df["Beguenstigter / Auftraggeber"]=df["Auftraggeber / Beguenstigter"]

        df["Betrag (EUR)"]=[x.replace(".","").replace(',', '.') for x in df['Betrag (EUR)']]
        mapped_df["Betrag"]=df["Betrag (EUR)"].astype(float)
        mapped_df["Buchungstag"]=pd.to_datetime(df["Buchungstag"],format="%d.%m.%Y")
        mapped_df["Verwendungszweck"]=df["Verwendungszweck"]
        mapped_df["Umsatzart"]=df["Buchungstext"]
        mapped_df["IBAN / Kontonummer"]=df["Kontonummer"]

        print("DKB Mapping successful: No error found..")
        data.enricher_input=mapped_df.to_csv()
        data.mapping_status="OK"
        print("DKB data mapped!")
        return True
    except Exception as e:
        print("this was no DKB file\n"+str(e))
        return False

def enrich_data(data):
    if data.mapping_status != "OK":
        data.result = json.dumps({"metadata":{"status":"FAILED"}})
        print("status: FAILED!")
    else:
        content = StringIO(data.enricher_input)
        try:
            filtered_df = pd.read_csv(content,sep=",")
            filtered_df["content"]=filtered_df["Beguenstigter / Auftraggeber"]+" "+filtered_df["Verwendungszweck"]
            
            data.processed = filtered_df.to_csv()
            data.categorizer_input = filtered_df.to_csv()
            data.processingTimestamp = datetime.now().strftime("%I:%M%p on %B %d, %Y")
            data.nrTransactions = str(len(filtered_df))
            print("[INFO] enrich data done!")
        except Exception as e:
            print("[ERROR] enrich data failed!",e)
        data.result = ""
        return data

def categorize_data(data,categorizer):
    action=categorizer.action
    model_path="categorizer_model"

    if(action=="train"):
        data=json.loads("rules")
        m=LogisticWord2VecModel()
        print("Training Model..")
        m.train(data)

        print("Persisting Model..")
        m.persist(model_path)

    if(action=="predict"):
        print(str(datetime.now()))
        print("Loading model..")
        input_data=data.categorizer_input
        m=LogisticWord2VecModel(model_path)
        print(str(datetime.now()))
        content=StringIO(input_data)
        input_data=pd.read_csv(content)
        print("Predicting categories..")
        #print(input_data.to_string())
        prediction=m.predict(input_data["content"])
        categorizer.output = pd.concat([input_data,prediction],axis=1).to_csv()
    return data

def cs_engines_purpow_compute(data,categorizer):
    from base64 import b64decode
    import random

    res=categorizer.output


    ### income ###
    content=StringIO(res)
    df=pd.read_csv(content)

    df["Buchungstag"]=pd.to_datetime(df["Buchungstag"])

    df["month"]=df["Buchungstag"].apply(lambda d:d.month)
    df["year"]=df["Buchungstag"].apply(lambda d:d.year)
    df["dow"]=df["Buchungstag"].apply(lambda d:d.dayofweek)


    #duration
    #amount_income = df[(df["Lohn"]==1)].groupby(["year","month"])["Betrag"].sum().mean()
    overall_income= df[(df["Lohn"]==1) & (df["Betrag"]>0.)]["Betrag"].sum()
    months=len(df.groupby(["year","month"]))
    amount_income=overall_income/months

    print(str(df["Buchungstag"]))
    print(str(df.groupby(["year","month"]).groups))
    print("overall income:"+str(overall_income))
    print("income: "+str(amount_income))
    print("Computing income indicator..")

    ### expenses ###

    result = -df[(df["Miete"]==1) &(df["Betrag"]<0)].groupby(["year","month"])["Betrag"].sum().mean()

    print("expenses: "+str(result))
    print("Computing expense indicator..")

    purpow = amount_income - result

    print("Purchasing Power: "+str(purpow))

    res={"purchasing_power":{
                "label":"Purchasing Power",
                "value":str(purpow),
                "type":"numeric",
                "description":"Balance available after deduction of monthly expenses.",
                "property":"purchasing power"
            },
             "income":{
               "label":"Income",
               "value":str(amount_income),
               "type":"numeric",
               "description":"Monthly net income",
               "property":"income",
             },
             "expenses":{
               "label":"Expenses",
               "value":str(result),
               "type":"numeric",
               "description":"Regular monthly expenses such as rent, incidental costs, car, insurance, etc.",
               "property":"expenses"
             }
        }
    data.indicators["purpow"] = json.dumps(res)
    return purpow

def aggregate_data(data):
    with open("distribution_data","r") as f:
        dists = json.loads(f.read())
    excluded_fields = []
    keys=list(data.indicators.keys())
    props=data.indicators

    indicators={}
    for i in range(len(keys)):
        key=str(keys[i])
        if(str(keys[i]) not in excluded_fields):
            val=props[key]

            #add distribution data..
            engine_entry=json.loads(val)
            for indicator_name,indicator in engine_entry.items():
                indicator["bins"]=dists[indicator_name]["bins"]
                indicator["values"]=dists[indicator_name]["values"]
                indicator["mu"]=dists[indicator_name]["mu"]
                indicator["std"]=dists[indicator_name]["std"]


                # compute indicators
                if(indicator["type"]=="numeric"):
                    bins=np.array(indicator["bins"][1:])
                    fraction=bins<=float(indicator["value"])
                    values=np.array(indicator["values"])
                    p=np.sum(values[fraction])/np.sum(values)
                    indicator["quantile"]=p
                else:
                    indicator["quantile"]=values[indicator["value"]]/np.sum(values)
                indicator["relevance"]=np.max([p,1-p])
                indicators.update(engine_entry)


    res={"metadata":{"status":"OK","processing_date":data.processingTimestamp},"indicators":indicators}      
    res=json.dumps(res)


    #pltool.setContextVar(libs.getLibObject("workspaces.CommonSign"),"tmp_resulting",res)
    data.tmpResult = res

    return data

def generate_report(data):
    aggregate = json.loads(data.tmpResult)

    entries = []
    for key, indicator in list(aggregate["indicators"].items())[:10]:
        entries += [{"name": key,"value": str(indicator["value"]),"description": indicator["description"],"Quantile": indicator["quantile"],"type": indicator["type"]}]

    data.Data = json.dumps(entries, indent=2)
    data.name = "cool name"
    data.nr_indicators = str(len(entries))

    name= ''.join(random.choices(string.ascii_uppercase + string.digits, k=12))

    #id=reporting.generateReport(nodeExecutionContext,"/data/IndicatorReport.jrxml",name).split(";")[0]
    #url="https://dms.commonsign.com:8443/alfresco/download/direct/workspace/SpacesStore/"+id+"/"+name+".pdf"

    aggregate["report"]={"url":"http://www.google.de"}

    data.tmp_result = json.dumps(aggregate)
    data.tmp_resulting = json.dumps(aggregate)
    return None

def generate_ui_output(data):
    # only map individual elements
    # plus map personality indicators single
    x = [1000, 2000, 3000]
    y = [10, 20, 30]



    json_string=data.tmp_result
    dat=json.loads(json_string)

    ui=map_ui(json_string)
    res = {"metadata": dat["metadata"],
           "report": dat["report"],
           "ui": ui,
           "indicators":dat["indicators"]}

    data.result = json.dumps(res)
    return data

def linear_interpolate(x, y, xp):
    """

    :param x: array of x values
    :param y: array of y values
    :param xp: value to interpolate for
    :return:
    """
    print("X vals"+str(x))
    print("X vals"+str(y))
    print("X vals"+str(xp))
    y=np.append(0,y)
    
    x, y = np.array(x), np.array(y)
    x_part1 = x[x <= xp]
    x_part2 = x[x > xp]

    y_part1 = y[x <= xp]
    y_part2 = y[x > xp]

    if(len(x_part1)==0):
        target_val=0.
        idx_i=0.
        xp=0.
    else:
        target_val = x_part1[-1]
        idx_i = np.where(x == target_val)[0][0]
        
    interp=y[-1]#max(y)
    if(x[-1]<=xp): # xp out of distribution range
        print("Rand ")
        return (xp, interp), ((np.append(x,xp).tolist(), (np.append(y, interp)).tolist())), ((np.append(xp,[])).tolist(),(np.append(interp,[])).tolist())
    

    # linear interpolation
    yp = y[idx_i] + ((y[idx_i + 1] - y[idx_i]) /
                     (x[idx_i + 1] - x[idx_i])) * (xp - x[idx_i])
    return (xp, yp), ((np.append(x_part1, xp).tolist(), (np.append(y_part1, yp)).tolist())), ((np.append(xp, x_part2)).tolist(),(np.append(yp, y_part2)).tolist())

def value_fct(x):
    if(x[1]["property"]=="none"):
        return -1.
    if(all(isinstance(e,str) for e in x[1]["bins"])):
        return 100000.
    else:
        return max(x[1]["bins"])

def map_ui(input_json):
    aggregate = json.loads(input_json)
    res = {}
    chart_nr = 1
    
    indicators=aggregate["indicators"]
    
    
    indicators=dict(sorted(indicators.items(),key=lambda x:x[1]["relevance"] if x[1]["relevance"]!=1 else 0.,reverse=True)[:7])
    
    
    #indicator_selection=#["tech_affinity","impulsivity","luxury_affinity","hedo","bummel"]
    indicator_selection=list(indicators.keys())
    #indicator_selection.remove("purchasing_power")
    #indicator_selection.remove("expenses")
    #indicator_selection.remove("income")
    
    
    #add vector indicator
    quantiles=[float(indicators[indicator]["quantile"]) for indicator in indicator_selection]
    bins=[indicators[indicator]["label"] for indicator in indicator_selection]
    
    #ground_truth=[indicators[indicator]["values"] for indicator in indicator_selection
    
    
    vector={"vec":{"values":quantiles,
        "bins":bins,
        "relevance":2.,
        "quantile":2.,
        "ground_truth":[0.5]*len(quantiles),
        "type":"vector",
        "description":"Pretty neat features",
        "property":"none"
        }}
    indicators.update(vector)
    
    low_props=sorted([v for  k,v in indicators.items() if v["quantile"]<=0.3 and v["property"]!="none"],key=lambda x:x["quantile"],reverse=True)
    high_props=sorted([v for k,v in indicators.items() if v["quantile"]>=0.7 and v["property"]!="none"],key=lambda x:x["quantile"],reverse=True)
            
    indicators=sorted(indicators.items(),key=value_fct)
    
    for key, indicator in indicators:
        print(key)
        print(str(indicator["bins"]))
        #pltool.printLine(str(indicator["values"]))
        print(str(indicator["quantile"]))
        
        el = {}
        type = indicator["type"]
        # Indicators info data..
        el["info"] = {"name": key,"description": indicator["description"],"type":indicator["type"]}
        if (type == "categorical"):
            el["data"] = [{"x":[-1,1],"y":[-1,1],"type":"scatter","opacity":0,"hoverinfo":"none","mode":"lines","line":{"color":"rgbba(0,0,0,0)"}},{"labels": indicator["bins"],"values": indicator["values"],"type": "pie","rotation":-indicator["values"][1] / 2 / np.sum(indicator["values"]) * 360,"opacity":0.5,"hole":0.2,"domain":{"x":[0.2,0.8],"y":[0.2,0.8]},"pull":[0.03,0.03],"text":indicator["bins"],"textposition":"outside","hoverinfo":"label+name","labels":["Value: {}<br>Quantile: {}".format(bin,quantile) for bin,quantile in zip(indicator["bins"],indicator["values"])],"name":"Peer","textfont":{"color":'rgba(255,255,255,1)'}},{"x":[0,0],"y":[0,1],"type":"scatter","hoverinfo":"text+name","text":"Value: 'Yes'<br>Quantile: "+str(indicator["quantile"])+"%","name":"You","mode":"lines","line":{"color":"rgba(10,180,130,1)"}}]
            el["configs"]={"editable":False,"scrollZoom":False}
            el["layout"]={"xaxis":{"title":indicator["label"],"titlefont":{"color":'rgba(250,250,250,0.45)'}},"yaxis":{"showgrid":False}}
        elif (type == "numeric"):
            xp = float(indicator["value"])
            _, part1, part2 = linear_interpolate(indicator["bins"], indicator["values"], xp)
            el["data"] = [{"x": part2[0],"y": part2[1],"type": "scatter","fill": "tozeroy","mode": "lines","hoverinfo": "none","line": {"width": 0,"shape": "spline","smoothing": 1}}, {"x": part1[0],"y": part1[1],"fill": "tozeroy","mode": "lines","type": "scatter","hoverinfo": "none","line": {"width": 0,"shape": "spline","smoothing": 1}}, {"x": [xp, xp],"y": [0,1.4*max(indicator["values"])],"type": 'scatter',"hoverinfo":"text+name","text":"Value: "+str("%.2f" % round(xp,2))+"<br>Quantile: "+str("%.2f" % round(indicator["quantile"],2)),"mode": "lines","line": {"color": 'rgba(10,180,130,1)'},"name": "You"}]
            el["configs"]={"editable":False,"scrollZoom":False}
            max_x=np.array(indicator["bins"][1:])[np.cumsum(indicator["values"])/sum(indicator["values"])>0.95]
            if(len(max_x)>0):
                max_x=max_x[0]
            else:
                max_x=max(indicator["bins"])
            max_x=max(max_x,xp*1.2)
            max_x=max(max_x,indicator["bins"][2])
            #"dtick":0.2,
            el["layout"]={"xaxis": { "showgrid": True,"range":[0,max_x],"title":indicator["label"],"domain": [0.17,0.83],"showticklabels":True,"tickmode":'auto',"ticks":'outside',"tick0":0,"nticks":6,"ticklen":0,
                "tickwidth":0,
                "gridcolor":'rgba(250,250,250,0.15)',
                "zerolinecolor":'rgba(250,250,250,0.45)',
                "tickfont":{"color":'rgba(250,250,250,0.45)'},
                "titlefont":{"size": 12,"color":'rgba(250,250,250,0.45)'},
                "tickcolor":'rgba(250,250,250,0.45)',
                "tickangle":0,
                #"autorange":True,
                "automargin":True
            },"yaxis": { "showgrid": True,
                "title":"Distribution",
                "range":[0.,1.1*max(indicator["values"])],
                "domain": [0.17,0.83],            
                "showticklabels":True,                
                "tickmode":'auto',
                "ticks":'outside',
                "tick0":0,
                "nticks":6,
                #"dtick":10,
                "ticklen":0,
                "tickwidth":0,
                "gridcolor":'rgba(250,250,250,0.15)',
                "zerolinecolor":'rgba(250,250,250,0.45)',
                "tickfont":{"color":'rgba(250,250,250,0.45)'},
                "titlefont":{"size": 12, "color":'rgba(250,250,250,0.45)'},
                "tickcolor":'rgba(250,250,250,0.45)',
                #"autorange":True,
                "automargin":True}}
        elif type == "vector":
            el["data"] = [{
                "type": 'scatterpolar',
                "r": indicator["ground_truth"],
                "theta": indicator["bins"],
                "fill": 'toself',
              "linewidth" : 4,
                "hoverinfo":"name+text",
              "hovertext":['Quantile: 49%', 'Quantile: 50%', 'Quantile: 58%', 'Quantile: 47%', 'Quantile: 48%', 'Quantile: 49%', 'Quantile: 50%'],
                "name": 'Peer group'
            }, {
                "type": 'scatterpolar',
                "r": indicator["values"],
                "theta": indicator["bins"],
                "fill": "toself",
              "linewidth" : 4,
              "hoverinfo":"name+text",
              "hovertext":['Quantile: 49%', 'Quantile: 50%', 'Quantile: 58%', 'Quantile: 47%', 'Quantile: 48%', 'Quantile: 49%', 'Quantile: 50%'],
                "name": "Your spectrum"
            }]
            el["layout"]={"polar":{
               "radialaxis": {
               "visible": True,
               "range": [0, 1.2*max([max(indicator["values"]),max(indicator["ground_truth"])])],
               "color": 'rgba(250,250,250,0.45)',
               "nticks":9,
               },
               "angularaxis": {
               "visible": True,
               "range": [0, 7],
               "orientation": -90,
                "color": 'rgba(250,250,250,0.45)',
              },
              "domain": { "x": [0.21,0.79], "y": [0.21,0.79]},
               "bgcolor" : "rgba(0, 0, 0,0)"}}

        res["summary-dashboard-chart-{}".format(chart_nr)] = el
        chart_nr += 1
        if (chart_nr > 8):
            break
    
    text=""
    

            
    if(len(high_props)>1):
        text+='Compared to your peer group, you show high values of '+'<span>'+high_props[0]["property"]+'</span>, '+'<span>'+high_props[1]["property"]+'</span>.'
       
    if(len(high_props)>3):
        text+='Additionally, you have high values of <span>'+high_props[2]["property"]+"</span> and <span>"+high_props[3]["property"]+"</span>.<br>"

    if(len(low_props)>2):
        text+='On the other hand, you show lower values of '+'<span>'+low_props[2]["property"]+'</span>, '+'<span>'+low_props[1]["property"]+'</span> and <span>'+low_props[0]["property"]+"</span>."
      


    res["SummaryDescription"]=text
    
    return {"Summary-Dashboard": res}


##################################### composite functions #####################################

def map_data(data):
    if map_DB(data):
        return data
    elif map_DKB(data):
        return data
    else:
        print("Data format not valid!")
        return None

def generate_output(data):
    generate_report(data)
    generate_ui_output(data)
    return None

def compute_indicators(data,categorizer):
    #cs_engines_lux_compute()
    cs_engines_purpow_compute(data,categorizer)
    """#cs_engines_travel_compute()
    cs_engines_technic_compute()
    cs_engine_pet()
    cs_engine_hedonism()
    cs_engine_bummel()
    cs_engines_debt()
    cs_engines_busy()
    cs_engines_impulsivity()
    #cs_meta_indicator()"""
    return data

def commonsignEngine(file_name):
    varData = data(file_name)
    categorizer = catEngine()
    dataInput = map_data(varData)

    # Enrich data
    dataEnriched = enrich_data(dataInput)

    # Categorization Engine
    dataCategorized = categorize_data(dataEnriched,categorizer)

    # Indicator Computation
    computedIndicators = compute_indicators(dataCategorized,categorizer)

    # Aggregate Engine Data
    data_aggregated = aggregate_data(computedIndicators)


    ######## create output ########
    generate_output(data_aggregated)
    return data_aggregated

##################################### a class for the input data #####################################
class data:
    def __init__(self,input_string):
        #try:
        #    self.input = pd.read_csv(file_path,sep=",")
        #except:
        #    self.input = pd.read_csv(file_path,sep=";")
        #self.input=StringIO(input_string)
        self.input = input_string
        self.result = None
        self.tmpResult = None
        self.callTimestamp = None
        self.processingTimestamp = None
        self.content = None
        self.prossed = None
        self.categorizer_input = None
        self.nrTransactions = None
        self.indicatorValue = None
        self.distributionData = None
        self.enricher_input = None
        self.mapping_status = None
        self.indicators = {}
        self.data = None
        self.name = None
        self.nr_indicators = 0
        self.tmp_result = None
        self.tmp_resulting = None

class catEngine:
    def __init__(self):
        self.rules = None
        self.action = "predict"
        self.output = None
        
