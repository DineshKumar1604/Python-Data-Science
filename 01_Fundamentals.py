# ===================================================================
# 01_python_fundamentals.py
# Description: Covers basic Python syntax, data structures, functions,
# error handling, and functional programming concepts.
# ===================================================================

print("--- SECTION 1: PYTHON FUNDAMENTALS ---")

# --- Basic Operations & Control Flow ---
print("\n--- Simple Calculator Example ---")
def add(a, b):
    return a + b

def diff(a, b):
    return a - b

def mul(a, b):
    return a * b

def div(a, b):
    if b == 0:
        return "Error: Cannot divide by zero!"
    return a / b

print(f"add(5, 3) = {add(5, 3)}")
print(f"div(10, 2) = {div(10, 2)}")


# --- Lists: Ordered, mutable collections ---
print("\n--- List Operations ---")
fruits = ["apple", "banana", "cherry"]
print(f"Original list: {fruits}")
fruits.append("orange") # Add an item
print(f"After appending 'orange': {fruits}")
fruits.remove("apple") # Remove an item
print(f"After removing 'apple': {fruits}")
print(f"Accessing an element by index (fruits[1]): {fruits[1]}")
print(f"Slicing the list (fruits[1:3]): {fruits[1:3]}")


# --- Dictionaries: Unordered, mutable key-value pairs ---
print("\n--- Dictionary Operations ---")
student = {"name": "John", "age": 25, "courses": ["Math", "Science"]}
print(f"Original dictionary: {student}")
print(f"Accessing a value by key (student['name']): {student['name']}")
student["age"] = 26 # Update a value
student["grade"] = "A" # Add a new key-value pair
print(f"After updating age and adding grade: {student}")
del student["age"] # Delete a key-value pair
print(f"After deleting age: {student}")

print("\nIterating through dictionary items:")
for key, value in student.items():
    print(f"{key}: {value}")


# --- Sets: Unordered, mutable, unique elements ---
print("\n--- Set Operations ---")
numbers = {1, 2, 3, 4, 5}
numbers.add(6)
print(f"Set of numbers: {numbers}")
evens = {2, 4, 6, 8}
print(f"Set of evens: {evens}")
union_set = numbers.union(evens)
print(f"Union of sets: {union_set}")
intersection_set = numbers.intersection(evens)
print(f"Intersection of sets: {intersection_set}")


# --- Functions and Palindrome Check ---
print("\n--- Function Example: Palindrome Check ---")
def is_palindrome(s):
    """Checks if a string is a palindrome."""
    return s == s[::-1]

s = "nitin"
if is_palindrome(s):
    print(f"'{s}' is a palindrome.")
else:
    print(f"'{s}' is not a palindrome.")


# --- Recursive Function: Factorial ---
print("\n--- Recursive Function: Factorial ---")
def factorial(n):
    """Calculates factorial recursively."""
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

print(f"The factorial of 5 is: {factorial(5)}")


# --- Error Handling with try-except-finally ---
print("\n--- Error Handling ---")
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Caught an error: {e}")
finally:
    print("This block always executes.")

try:
    # This will raise a FileNotFoundError
    with open("non_existent_file.txt", 'r') as file:
        content = file.read()
except FileNotFoundError as e:
    print(f"Caught a file error: {e}")


# --- Lambda Functions, Map, and Filter ---
print("\n--- Lambda, Map, Filter ---")
numbers_list = [1, 2, 3, 4, 5]
# Map: applies a function to all items in an iterable
squared_numbers = list(map(lambda x: x**2, numbers_list))
print(f"Squared numbers using map: {squared_numbers}")

# Filter: creates a list of elements for which a function returns true
even_numbers = list(filter(lambda x: x % 2 == 0, numbers_list))
print(f"Even numbers using filter: {even_numbers}")


# --- List Comprehensions ---
print("\n--- List Comprehensions ---")
squares = [x**2 for x in range(10)]
print(f"Squares from 0 to 9: {squares}")

even_squares = [x**2 for x in range(10) if x % 2 == 0]
print(f"Squares of even numbers: {even_squares}")

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat_list = [num for row in matrix for num in row]
print(f"Flattened matrix: {flat_list}")

print("\n--- END OF FUNDAMENTALS SCRIPT ---")
