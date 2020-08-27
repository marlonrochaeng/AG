from Population import Population
import time

qtd_exec = 100
#instanciando e criando a 1 pop.
p = Population()
#marcando o tempo de execucao
start_time = time.time()
for i in range(qtd_exec):
    p.create_first_gen()
    #gerando 50 novas populacoes atraves de crossover, mutacao e roleta
    p.generate_n_pop(50)
#marcando o fim da execucao
end_time = time.time()
total = end_time - start_time
print(f"--- {total} segundos se passaram ao executar {qtd_exec} vezes o problema---\n")