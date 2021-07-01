import pandas as pd
import csv

def read_data(filename=''):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        headers = []
        rows = []
        for row in csv_reader:
            if line_count == 0:
                headers += row
                line_count += 1
            else:
                rows.append(row)
                line_count += 1

    data = pd.DataFrame(rows, columns = headers)

    return data


def init_conditional_features(data, table, features = []):
    if len(features) == 1:
        ftr = features[0]
        table[ftr] = data[ftr].unique()

    elif len(features) == 2:
        ftr0 = features[0]
        ftr1 = features[1]
        col0_data = []
        col1_data = []
        for val0 in data[ftr0].unique():
            for val1 in data[ftr1].unique():
                col0_data.append(val0)
                col1_data.append(val1)
        #conditional features
        table[ftr0] = col0_data
        table[ftr1] = col1_data
    
    return table

def prob(data, feature, value):
    freq = 0
    for index, row in data.iterrows():
        if row[feature] == value:
            freq += 1
    return freq / len(data)

def conditional_prob(data, conditions = [], con_vals = [], feature = '', value = ''):
    con_freq = 3
    freq = 6
    for index, row in data.iterrows():
        find = True
        for i in range(len(conditions)):
            if row[conditions[i]] != con_vals[i]:
                find = False
        if find:
            freq += 1
        if find and row[feature] == value:
            con_freq +=1
    return con_freq / freq

def create_table(data, feature, conditional_features = []):
    table = pd.DataFrame()
    #find diff values in conditional columns
    table = init_conditional_features(data, table, conditional_features)
    #find feature cols
    feature_uniques = data[feature].unique()
    for val in feature_uniques:
        table[val] = 0.0
    
    #calc probs
    for val in feature_uniques:
        if len(conditional_features) == 0:
            pr = prob(data,feature,val)
            table[val] = [pr]
        else:
            for index, row in table.iterrows():
                con_vals = [row[ftr] for ftr in conditional_features]
                table.at[index, val] = conditional_prob(data, conditional_features, con_vals, feature, val)

    print(feature,'table')
    print(table)
    print()
    
    return table

def main():
    data = read_data('Final-Train.txt')

    #Network Connections based on Network.PNG and Descripton.pdf
    table_Overall_Score = create_table(data, 'Overall_Score')
    table_Price = create_table(data, 'Price', ['Overall_Score','Safety'])
    table_Doors = create_table(data, 'Doors',['Overall_Score','Seating_Capacity'])
    table_Luggage_Size = create_table(data, 'Luggage_Size',['Overall_Score'])
    table_Seating_Capacity = create_table(data, 'Seating_Capacity', ['Overall_Score','Safety'])
    table_Maintenance_Costs = create_table(data, 'Maintenance_Costs', ['Overall_Score', 'Safety'])
    table_Safety = create_table(data, 'Safety', ['Overall_Score', 'Luggage_Size'])

    test_data = read_data('Final-Test.txt')

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    
    print('TEST STARTED... ')
    print('Overall_Score Values:')
    print('')
    for index,test in test_data.iterrows():
        
        Price = test['Price']
        Maintenance_Costs = test['Maintenance_Costs']
        Doors = test['Doors']
        Seating_Capacity = test['Seating_Capacity']
        Luggage_Size = test['Luggage_Size']
        Safety = test['Safety']
        _Overall_Score = test['Overall_Score']

        result = {}
        for col in table_Overall_Score.columns:
            p_test = 1.0
            p_test *= table_Overall_Score[col].values[0]
            p_test *= table_Price.loc[(table_Price['Overall_Score'] == col) 
                                    & (table_Price['Safety'] == Safety),Price].values[0]
            p_test *= table_Maintenance_Costs.loc[(table_Maintenance_Costs['Overall_Score'] == col)
                                    & (table_Maintenance_Costs['Safety'] == Safety),Maintenance_Costs].values[0]
            p_test *= table_Doors.loc[(table_Doors['Overall_Score'] == col)
                                  & (table_Doors['Seating_Capacity'] == Seating_Capacity),Doors].values[0]
            p_test *= table_Seating_Capacity.loc[(table_Seating_Capacity['Overall_Score'] == col) 
                                    & (table_Seating_Capacity['Safety'] == Safety),Seating_Capacity].values[0]
            p_test *= table_Safety.loc[(table_Safety['Overall_Score'] == col)
                                    & (table_Safety['Luggage_Size'] == Luggage_Size),Safety].values[0]
            p_test *= table_Luggage_Size.loc[(table_Luggage_Size['Overall_Score'] == col),Luggage_Size].values[0]

            result[col] = p_test
            print(_Overall_Score, '\t', col, '\t', p_test)
        value = -1
        if result['bad'] != 0 and result['good'] != 0:
            value = result['bad']/(result['bad']+result['good'])

        if value >= 0.5:
            if _Overall_Score == 'bad':
                true_positive += 1
            else:
                false_positive += 1
        else:
            if _Overall_Score == 'good':
                true_negative += 1
            else:
                false_negative += 1
    accuracy = (true_positive+true_negative)/test_data.shape[0]
    #another way to find acc
    #accuracy = (true_positive+true_negative)/(true_positive+true_negative+false_positive+false_negative)
    tp_rate = true_positive/(true_positive+false_negative)
    tn_rate = true_negative/(false_positive+true_negative)
    print('RESULTS: ','\n',
          'TP= ', true_positive ,'\n',
          'TN= ', true_negative,'\n',
          'FP= ', false_positive,'\n',
          'FN= ', false_negative,'\n',
          'TP Rate= ', tp_rate,'\n',
          'TN Rate= ', tn_rate,'\n',
          'Accuracy= ', accuracy)

main()
