from Population import Population

p = Population()

p.create_first_gen()


print("=================================\n")
print("Occurrences:\n")
print(p.get_letter_occurrence(0))
print("=================================\n")
print("Population len:\n")
print(len(p.pop))
print("Population:\n")
print(p.pop)
print("=================================\n")
print("Population after crossover:\n")
p.add_crossoved_gen()
print("Population len:\n")
print(len(p.pop))
print(p.pop)
print("=================================\n")
print(p.order_pop())
print(p.pop)