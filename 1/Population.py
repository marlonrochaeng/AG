import random
import time
#sendmory
class Population:

  def __init__(self):
    self.pop = []

  def get_fitness(self, i):
    return abs((i[0]*1000 + i[1]*100 + i[2]*10 + i[3] + i[4]*1000 + i[5]*100 + i[6]*10 + i[1]) - (i[4]*10000 + i[5]*1000 + i[2]*100 + i[1]*10 + i[-1]))

  def create_first_gen(self):
    for i in range(100):
      individuo = [random.randint(0,9) for f in range(8)]
      self.pop.append((individuo,self.get_fitness(individuo)))
  
  def get_ordered_pop(self):
    return sorted(self.pop, key = lambda x: x[1])

  def order_pop(self):
    self.pop.sort(key = lambda x: x[1])
  
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

  def update_to_best_individuals(self):
    self.pop = self.pop[:70]

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

  def generate_n_pop(self, n):
    start_time = time.time()
    for i in range(n):
      self.add_crossoved_gen()
      self.mutation()
      self.generate_new_pop()
    end_time = time.time()
    total = end_time - start_time
    print(f"--- {total} segundos se passaram ao gerar 100 novas populacoes---\n")

  def generate_new_pop(self):
    new_pop = []
    for i in range(100):
      self.order_pop()
      new_pop.append(self.pop.pop(self.roulette_wheel()))

    self.pop = new_pop

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

  def add_crossoved_gen(self):
    parents = self.pop.copy()
    while len(parents) > int(len(self.pop)/3):
        p1 = parents.pop(random.randint(0,len(parents)-1))
        p2 = parents.pop(random.randint(0,len(parents)-1))
        aux1, aux2 = self.cycle_cross(p1[0], p2[0])
        self.pop.append(aux1)
        self.pop.append(aux2)

  def get_positions(self):
    p1 = random.randint(0,7)
    p2 = random.randint(0,7)

    while p2 == p1:
      p2 = random.randint(0,7)
    
    return p1,p2

  def mutation(self):
    for i in range(len(self.pop)):
      if random.uniform(0,100) <= 3:
        p1, p2 = self.get_positions()
        aux = self.pop[i][0][p1]
        self.pop[i][0][p1] = self.pop[i][0][p2]
        self.pop[i][0][p2] = aux
        self.pop[i] = self.pop[i][0], self.get_fitness(self.pop[i][0])

