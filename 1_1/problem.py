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
p.add_crossoved_gen(80)
print("Population len:\n")
print(len(p.pop))
print(p.pop)
print("=================================\n")
print(p.order_pop())
print(p.pop)
p.mutation()
print("=================================\n")
print("New generation:")
p.generate_new_pop()
print(p.pop)
print("=================================\n")
print(p.get_ordered_pop()[:10])
print("=================================\n")
p.generate_n_pop(50)
print(p.get_ordered_pop()[:10])