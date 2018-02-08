from flask import render_template
from flask import Flask


import plotly as py
import plotly.graph_objs as go

pyplt = py.offline.plot

app = Flask(__name__)

@app.route('/')
def index():
    
    # Data
    data_1 = go.Bar(
                x = ["上海物贸", "广东明珠", "五矿发展"],
                y = [4.12, 5.32, 0.60],
                name = "201609"
        )
    
    data_2 = go.Bar(
                x = ["上海物贸", "广东明珠", "五矿发展"],
                y = [3.65, 6.14, 0.58],
                name = "201612"
        )
    
    data_3 = go.Bar(
                x = ["上海物贸", "广东明珠", "五矿发展"],
                y = [2.15, 1.35, 0.19],
                name = "201703"
        )
    
    Data = [data_1, data_2, data_3]
    
    # Layout
    layout = go.Layout(
                title = '国际贸易板块净资产收益率对比图'
        )
    
    # Figure
    figure = go.Figure(data = Data, layout = layout)
    
    div = pyplt(figure, output_type='div', auto_open=False, show_link=False)
    context = {}
    context['graph'] = div

    import sys
    print('参数div占用内存大小为 %d bytes'%sys.getsizeof(div))
    with open('div1.txt', 'w') as file:
        file.write(div)
        

    return render_template("index2.html",
        title = 'Home',
        context = context)   


if __name__ == '__main__':
      app.run()     


 