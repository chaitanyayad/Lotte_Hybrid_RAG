# Qualitative RAG Evaluation Check (Day 2 Deliverable)

Comparison of RAG answers when gold passages are present vs missing from top-5 retrieval:

### Query 0: *"how much should i feed my 1 year old english mastiff?"*
- **Gold PIDs**: `[2466, 2435, 1641, 4619, 1615]`
- **Condition A (Gold Present)**:
  - **Answer**: I have a 2 1/2 year old bull mastiff. [1] Each of my English mastiffs eat that in a 2 week period, so each mastiff will cost you just on food $120 a month. [2]
  - **Citations Valid**: Yes (Cites: [1, 2])
- **Condition B (Gold Missing)**:
  - **Answer**: If you dig around the internet, you will find out that English Bull Mastiffs have a list of health issues: Hip dysplasia This affects almost 1/4th of the mastiff population. [3] My 4 year old mostly speaks English (as much as he speaks anything) but does have a bunch of Hebrew words. [4]
  - **Says 'not found'**: No
- **Observation**: Accurately answers and cites sources when gold passage is present; attempts partial answer from distractor text.

---

### Query 1: *"are zebra loaches safe with shrimp?"*
- **Gold PIDs**: `[3811, 2053, 619, 7323, 4573]`
- **Condition A (Gold Present)**:
  - **Answer**: I made the mistake of keeping Kouli Loaches in a 55 gal heavily planted tank with Amano shrimp. [1] I thought they were just finding good hiding places until late one night I was watching and one of my loaches attacked and ate an Amano shrimp. [1]
  - **Citations Valid**: Yes (Cites: [1])
- **Condition B (Gold Missing)**:
  - **Answer**: As for using them as food, Im not sure itd work - if you fish dont actively eat the shrimp currently, then they are unlikely to be bothered about eating them once they are dead either as they are likely herbivores If you happen to have another tank that has loaches in (I know both Pakistani and Clown loaches will) they will eat shrimps and snails quite happily. [1] He happily shared his 3.5 gallon (13 liters) habitat with some ghost shrimp and a zebra snail. [3]
  - **Says 'not found'**: No
- **Observation**: Accurately answers and cites sources when gold passage is present; attempts partial answer from distractor text.

---

### Query 2: *"are serpae tetras fin nippers?"*
- **Gold PIDs**: `[3457, 8406, 2837, 2838]`
- **Condition A (Gold Present)**:
  - **Answer**: Serpae tetras are by nature fin nippers. [1] Serpae tetras tend to be less aggressive if kept in a larger group, like more than 6-8. [2]
  - **Citations Valid**: Yes (Cites: [1, 2])
- **Condition B (Gold Missing)**:
  - **Answer**: not found
  - **Says 'not found'**: Yes
- **Observation**: Accurately answers and cites sources when gold passage is present; appropriately returns 'not found' or refrains from unsupported answers when gold passage is absent.

---

### Query 3: *"do neon tetras eat shrimp?"*
- **Gold PIDs**: `[5159]`
- **Condition A (Gold Present)**:
  - **Answer**: Neon tetras will indeed eat baby shrimp. [1] Neon tetras will attempt to eat these. [2]
  - **Citations Valid**: Yes (Cites: [1, 2])
- **Condition B (Gold Missing)**:
  - **Answer**: Neon tetras will attempt to eat these. [1] Ive kept neon tetras with shrimp and they would often pursue the baby shrimp. [1]
  - **Says 'not found'**: No
- **Observation**: Accurately answers and cites sources when gold passage is present; attempts partial answer from distractor text.

---

### Query 4: *"how much should i feed my english mastiff?"*
- **Gold PIDs**: `[2466, 2435, 1641, 4619, 1615]`
- **Condition A (Gold Present)**:
  - **Answer**: Each of my English mastiffs eat that in a 2 week period, so each mastiff will cost you just on food $120 a month. [1] A mastiff is going to eat a lot, no matter what you feed him, but the better the dog food, the less he is going to need to eat. [3]
  - **Citations Valid**: Yes (Cites: [1, 3])
- **Condition B (Gold Missing)**:
  - **Answer**: If you dig around the internet, you will find out that English Bull Mastiffs have a list of health issues: Hip dysplasia This affects almost 1/4th of the mastiff population. [2] So why does The English Bull Mastiff, which is not even the biggest of all Mastiffs, have a shorter life span?. [2]
  - **Says 'not found'**: No
- **Observation**: Accurately answers and cites sources when gold passage is present; attempts partial answer from distractor text.

---

### Query 5: *"is cat skin tougher than human skin?"*
- **Gold PIDs**: `[5462]`
- **Condition A (Gold Present)**:
  - **Answer**: One of the main reasons why you shouldnt use human shampoo on a cat (or bathe it at all if its not absolutely necessary) is that human skin is more acidic than cat skin. [3] Here is a PDF link about dog and cat skin thickness. [1]
  - **Citations Valid**: Yes (Cites: [1, 3])
- **Condition B (Gold Missing)**:
  - **Answer**: One of the main reasons why you shouldnt use human shampoo on a cat (or bathe it at all if its not absolutely necessary) is that human skin is more acidic than cat skin. [2] Older cats have more loose skin on the belly than younger cats do, and I think female cats have a tiny bit more loose skin around the mammaries than male cats have in the same area. [1]
  - **Says 'not found'**: No
- **Observation**: Accurately answers and cites sources when gold passage is present; attempts partial answer from distractor text.

---

