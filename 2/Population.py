import random
import time
import numpy as np
#sendmory
class Population:

  def __init__(self):
    self.pop = []

  def get_fitness(self, i):
    return abs((i[0]*1000 + i[1]*100 + i[2]*10 + i[3] + i[4]*1000 + i[5]*100 + i[6]*10 + i[1]) - (i[4]*10000 + i[5]*1000 + i[2]*100 + i[1]*10 + i[-1]))

  def create_first_gen(self):
    self.pop = []
    for i in range(250):
      options = list(range(10))
      individuo = []
      while options:
        individuo.append(options.pop(options.index(random.choice(options))))
      self.pop.append((individuo,self.get_fitness(individuo)))
  
  def get_ordered_pop(self):
    return sorted(self.pop, key = lambda x: x[1])

  def order_pop(self):
    self.pop.sort(key = lambda x: x[1])

  def order_pop_reverse(self):
    self.pop.sort(key = lambda x: x[1],reverse=True)
  
  def get_letter_occurrence(self, i):
    occurrences = {}
    for p in self.pop:
      if occurrences.get(p[0][i]):
        occurrences[p[0][i]] = occurrences[p[0][i]] + 1
      else:
        occurrences[p[0][i]] = 1
    
    for o in occurrences:
      occurrences[o] = occurrences[o]/len(self.pop) * 100
    return occurrences

  def roulette_wheel(self):
    fs = [i[1] for i in self.pop]

    sum_fs = sum(fs)
    max_fs = max(fs)
    min_fs = min(fs)

    p = random.random()*sum_fs
    t = max_fs + min_fs

    choosen = 0

    for i in range(len(self.pop)):
      p -= (t - self.pop[i][1])
      if p < 0:
        choosen = i
        break
    
    return choosen

  def rouletteWheelSelect(self):
    fitnessSum = 1
    probs = [0 for i in self.pop]
    for individual in self.pop:
      fitnessSum += individual[1]
    for i in range(len(self.pop)):
      probs[i] = self.pop[i][1] / fitnessSum
    probSum = 0
    wheelProbList = []

    for p in probs:
      probSum += (1-p) / (len(self.pop)-1)
      wheelProbList.append(probSum)

    r = random.random()  # 0<=r<1
    for i, p in enumerate(wheelProbList):
      if r < p:
        return i
    return i

  def single_crossover(self, pos):
    p1_index, p2_index = self.rouletteWheelSelect(), self.rouletteWheelSelect()
    while p1_index == p2_index:
      p2_index = self.rouletteWheelSelect()
    temp1 = self.pop[p1_index][0][:pos] + self.pop[p2_index][0][pos:]
    temp2 = self.pop[p2_index][0][:pos] + self.pop[p1_index][0][pos:]
    child1 = temp1, self.get_fitness(temp1)
    child2 = temp2, self.get_fitness(temp1)

    return child1, child2

  def multi_crossover(self, pos):
    for i in pos:
      child1, child2 = self.single_crossover(i)

    return [(child1[0], self.get_fitness(child1[0])) , (child2[0], self.get_fitness(child2[0]))]

  def generate_n_pop(self, n):
    #start_time = time.time()
    for i in range(n):
      self.add_crossoved_gen(80)
      self.mutation()
      self.generate_new_pop()
    #end_time = time.time()
    #total = end_time - start_time
    #print(f"--- {total} segundos se passaram ao gerar {n} novas populacoes---\n")

  def generate_new_pop(self):
    new_pop = []
    self.order_pop()
    #for i in range(100):
    #  pos = self.linear_rank()
      #print("Pos:",pos)
      #print("Tam:",len(self.pop))
    #  new_pop.append(self.pop.pop(pos))
      #print("Position:",pos)
      #print("Individual:",self.pop[pos])

    #self.pop = new_pop
    self.pop = self.pop[0:250]
  
  def linear_rank(self):
    pos = []
    cum = []

    for i in range(len(self.pop),0,-1):
      pos.append(i)
      cum.append(sum([a for a in pos]))
    
    #print(pos)
    sel = random.choices(pos,  cum_weights=cum,k=1)
    #print("Sel:",sel)
    #print("Ind:",self.pop[sel[0]])
    return pos.index(sel[0])


  def add_crossoved(self, num):
    childs = []
    for i in range(40):
      c1, c2 = self.multi_crossover(self.get_positions())
      childs.append(c1)
      childs.append(c2)
    self.pop += childs
    print("New pop len after crossover:",len(self.pop))

  def create_new_individual(self):
    individual = []
    for i in range(8):
      occurrence = self.get_letter_occurrence(i)
      value = self.roulette_wheel(occurrence)
      individual.append(value)
    return individual
  
  def complete_population(self):
    for i in range(30):
      individual = self.create_new_individual()
      self.pop.append((individual,self.get_fitness(individual)))

  def cycle_cross(self, p1, p2):
    cycles = [-1]*len(p1)
    cycle_no = 1
    cyclestart = (i for i,v in enumerate(cycles) if v < 0)

    for pos in cyclestart:

        while cycles[pos] < 0:
            cycles[pos] = cycle_no
            if p2[pos] in p1:
                pos = p1.index(p2[pos])
            
        cycle_no += 1

    child1 = [p1[i] if n%2 else p2[i] for i,n in enumerate(cycles)]
    child2 = [p2[i] if n%2 else p1[i] for i,n in enumerate(cycles)]

    return (child1, self.get_fitness(child1)), (child2, self.get_fitness(child2))

  def cx_modified(self):
    #representation = p1
    #partner = p2

    index1, index2 = self.get_positions_simple(99)
    idx_start = random.randint(0, 7)
    idx_switchs = [idx_start]
    cycle_start = self.pop[index1][0][idx_start]
    current_value = self.pop[index2][0][idx_start]
    cond = True

    while cond:
      idx = self.pop[index1][0].index(current_value)
      current_value = self.pop[index2][0][idx]
      cond = current_value != cycle_start
      idx_switchs.append(idx)
    idx_switchs = list(set(idx_switchs))
    
    child_1 = self.pop[index1][0].copy()
    child_2 = self.pop[index2][0].copy()
    # print(idx_switchs)
    for i in idx_switchs:
      aux = child_1[i]
      child_1[i] = child_2[i]
      child_2[i] = aux

    return [(child_1, self.get_fitness(child_1)), (child_2, self.get_fitness(child_2))]

  def tournament(self):
    p1, p2 = self.get_positions(len(self.pop)-1)
    p3 = random.randint(0,99)
    while p3 == p1 or p3 == p2:
      p3 = random.randint(0,99)
    
    if self.pop[p1][1] < self.pop[p2][1] and self.pop[p1][1] < self.pop[p3][1]:
      return p1
    elif self.pop[p2][1] < self.pop[p1][1] and self.pop[p2][1] < self.pop[p3][1]:
      return p2
    else:
      return p3


  def pmx(self):
    tam = 10
    p1, p2 = [0] * tam, [0] * tam

    parent1, parent2 = self.rouletteWheelSelect(), self.rouletteWheelSelect()
    while parent1 == parent2:
      parent2 = self.rouletteWheelSelect()
    #parent1, parent2 = self.linear_rank(), self.linear_rank()
    #parent1, parent2 = self.tournament(), self.tournament()

    ind1 = self.pop[parent1][0].copy()
    ind2 = self.pop[parent2][0].copy()

    for i in range(tam):
      p1[self.pop[parent1][0][i]] = i
      p2[self.pop[parent2][0][i]] = i
    
    cxpoint1 = random.randint(0, tam)
    cxpoint2 = random.randint(0, tam - 1)

    if cxpoint2 >= cxpoint1:
      cxpoint2 += 1
    else:  # Swap the two cx points
      cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    for i in range(cxpoint1, cxpoint2):
      # Keep track of the selected values
      temp1 = ind1[i]
      temp2 = ind2[i]
      # Swap the matched value
      ind1[i], ind1[p1[temp2]] = temp2, temp1
      ind2[i], ind2[p2[temp1]] = temp1, temp2
      # Position bookkeeping
      p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
      p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return (ind1, self.get_fitness(ind1)), (ind2, self.get_fitness(ind2))

  def cx_cross_pmx(self):
    p1,p2 = self.get_positions(9)
    if p1 > p2:
      p1, p2 = p2, p1
    print(p1,p2)
    parent1, parent2 = self.get_positions(len(self.pop)-1)

    parent1_map = self.pop[parent1][0].copy()
    parent2_map = self.pop[parent2][0].copy()

    filho1 = [-1]*10
    filho2 = [-1]*10

    for i in range(p1,p2,1):
      filho1[i] = parent1_map[i]
      filho2[i] = parent2_map[i]
    
    for i in range(p1,p2,1):
      flag = True
      if filho2[i] not in filho1[p1:p2]:
        aux = i
        while flag:
          if parent1_map[aux] in parent2_map[p1:p2]:
            aux = parent2_map.index(parent1_map[aux])
          else:
            index = parent2_map.index(parent1_map[aux])
            print(parent2_map[aux])
            flag = False
            filho1[index] = filho2[i]

    for i in range(len(filho1)):
      if filho1[i] == -1:
        filho1[i] = parent2_map[i]

    for i in range(p1,p2,1):
      flag = True
      if filho1[i] not in filho2[p1:p2]:
        aux = i
        while flag:
          if parent2_map[aux] in parent1_map[p1:p2]:
            aux = parent1_map.index(parent2_map[aux])
          else:
            flag = False
            index = parent1_map.index(parent2_map[aux])         
            filho1[index] = filho1[i]

    for i in range(len(filho1)):
      if filho2[i] == -1:
        filho2[i] = parent1_map[i]
    print(parent1_map,filho1)
    print(parent2_map,filho2)
    
    return [(filho1, self.get_fitness(filho1)), (filho2, self.get_fitness(filho2))]

    

  def get_positions(self, num=7):
    p1 = random.randint(0,num)
    p2 = random.randint(0,num)

    while p2 == p1:
      p2 = random.randint(0,7)
    
    return p1,p2

  def get_positions_simple(self, num=7):
    p1 = random.randint(0,num)
    p2 = random.randint(0,num)
    
    return p1,p2  

  def add_crossoved_gen(self, num):
    aux_pop = []
    parents = self.pop.copy()
    self.order_pop_reverse()

    for i in range(num):
      
      #p1,p2 = self.linear_rank(), self.linear_rank()
      #p1,p2 = self.tournament(), self.tournament()
      #aux1, aux2 = self.cycle_cross(parents[p1][0], parents[p2][0])
      #aux1, aux2 = self.pmx()
      aux1, aux2 = self.multi_crossover(self.get_positions(9))
      aux_pop.append(aux1)
      aux_pop.append(aux2)
    self.pop += aux_pop

  def mutation(self):
    for i in range(len(self.pop)):
      if random.uniform(0,250) <= 25:
        p1, p2 = self.get_positions()
        aux = self.pop[i][0][p1]
        self.pop[i][0][p1] = self.pop[i][0][p2]
        self.pop[i][0][p2] = aux
        self.pop[i] = self.pop[i][0], self.get_fitness(self.pop[i][0])

