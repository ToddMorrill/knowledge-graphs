from openie import StanfordOpenIE

with StanfordOpenIE() as client:
    text = 'Barack Obama was born in Hawaii. Richard Manning wrote this sentence.'
    print('Text: %s.' % text)
    for triple in client.annotate(text):
        print('|-', triple)

    # graph_image = 'graph.png'
    # client.generate_graphviz_graph(text, graph_image)
    # print('Graph generated: %s.' % graph_image)