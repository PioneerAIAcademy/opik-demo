# BYU-Pathway Chatbot Answer Evaluation Grading Rubric

## Overview
This rubric is used to evaluate chatbot-generated answers against ideal answers, assessing how well the generated responses are grounded in retrieved content and how accurately they address the questions.

## Scoring Scale (1-5)

### Score 1 - Contradictory or Inappropriately Sourced
**Definition**: Generated answer contradicts the ideal answer OR uses external knowledge to provide wrong/inappropriate information

**Characteristics:**
- Answer provides information that directly contradicts the ideal answer
- Answer relies on external knowledge not present in the retrieved content to give incorrect information
- Answer fundamentally misunderstands the question or context
- Answer says "Yes" when ideal answer says "No" (or vice versa)
- Answer provides wrong values, facts, or procedures

**Examples from evaluation data:**
- "We have information that still says that a bachelor's degree lasts 120 credits" (when correct answer is 90-96 credits)
- "Uses external knowledge" (providing Canvas Zoom integration info not in retrieved content)
- "doesn't understand that friend != member" (misinterpreting Church membership requirements)
- Answer addresses wrong audience entirely (e.g., answering for students when question is for missionaries)

**Key indicator**: The answer is **WRONG**

---

### Score 2 - Not Grounded or Insufficient Content
**Definition**: Generated answer is not grounded in the retrieved content OR contains insufficient/missing content

**Characteristics:**
- Answer is not based on the retrieved content
- Answer extrapolates or makes up information not in the source material
- Retrieved content is insufficient to properly answer the question
- Answer addresses wrong audience or misinterprets context (but not as severely as Score 1)
- Answer includes fabricated details, procedures, or requirements

**Examples from evaluation data:**
- "Insufficient content"
- "missing content; extrapolating"
- "made up answer" (adding details about communication requirements, fairness considerations, etc. not in source)
- "Incorrect answer based on incorrect data/content available"
- Answer describes processes or features that don't exist in the system

**Key indicator**: The answer is **UNGROUNDED** (not based on retrieved content, but may not directly contradict)

---

### Score 3 - Partially Correct but Incomplete
**Definition**: Generated answer is partially correct but has significant gaps or limitations

**Characteristics:**
- Answer provides some relevant information but misses key details
- Answer refuses to answer when it could provide useful information
- Answer is in the right direction but significantly incomplete
- Answer addresses only part of the question
- Answer lacks important context or steps present in ideal answer

**Examples from evaluation data:**
- Answer says "Sorry, I'm not able to answer this question. Could you rephrase it?" when it could provide partial information
- "The retrieved content is good, only the response generated is not very precise"
- Answer provides general guidance but misses specific procedures or requirements

**Key indicator**: The answer is **PARTIALLY USEFUL** but has significant limitations

---

### Score 4 - Mostly Correct with Minor Issues
**Definition**: Generated answer is mostly correct and grounded in retrieved content but has minor issues with completeness, specificity, or grounding

**Characteristics:**
- Answer is generally accurate but lacks some specificity or completeness
- Answer is well-grounded but missing minor details present in ideal answer
- Answer is good but could be more precise or comprehensive
- Answer provides correct information but in less organized or less clear manner
- Answer omits non-critical but helpful details

**Examples from evaluation data:**
- "Content updates needed" (answer was correct based on old content)
- "The organization needs to provide some additional data/content to help further clarify the answer"
- "Only thing that would make this response better is to provide the link to the meeting house locator"
- Answer is correct but doesn't include all steps or minor details

**Key indicator**: The answer is **GOOD** but not perfect

---

### Score 5 - Semantically Equivalent
**Definition**: Generated answer is semantically equivalent to the ideal answer and properly grounded in retrieved content

**Characteristics:**
- Answer conveys the same meaning and information as the ideal answer
- Answer is properly grounded in the retrieved content
- Answer addresses the question completely and accurately
- Answer may use different wording but captures all essential information
- Answer is appropriate for the intended audience
- Answer includes all key steps, requirements, or details

**Examples from evaluation data:**
- Answers that received 5,5,5,5,5 scores across all evaluators
- Answers that accurately reflect policies, procedures, and requirements
- Answers that properly cite and use the retrieved content

**Key indicator**: The answer is **EXCELLENT** and meets all quality criteria

---

## Key Evaluation Criteria

### 1. Grounding
- Is the answer based on the retrieved content?
- Does the answer cite or reference information from the retrieved content?
- Does the answer avoid making up details not present in the source?

### 2. Accuracy
- Does the answer align with the ideal answer?
- Are facts, figures, and procedures correct?
- Does the answer avoid contradictions with the ideal answer?

### 3. Completeness
- Does the answer address all aspects of the question?
- Are all key steps, requirements, or details included?
- Does the answer provide sufficient information for the user to take action?

### 4. Relevance
- Does the answer stay focused on the question asked?
- Does the answer avoid tangential or unnecessary information?
- Is the level of detail appropriate for the question?

### 5. Audience Appropriateness
- Does the answer address the correct audience (students vs. missionaries vs. administrators)?
- Is the language and tone appropriate for the intended audience?
- Does the answer reference the correct resources and tools for that audience?

---

## Scoring Guidelines

### When scores diverge among evaluators:
- Different evaluators may weigh criteria differently (e.g., one prioritizes grounding while another prioritizes completeness)
- Borderline cases between scores (especially 1 vs 2, or 4 vs 5) may receive mixed ratings
- Average scores help smooth out individual evaluator variance

### Common scoring patterns:
- **All 1s**: Clear contradiction or wrong answer using external knowledge
- **Mix of 1s and 2s**: Answer has problems with both accuracy and grounding
- **All 2s**: Answer is clearly ungrounded but doesn't directly contradict
- **Mix of 2s and 4s**: Answer has grounding issues but gets some things right
- **All 5s**: Answer is clearly equivalent to ideal answer

### The distinction between Score 1 and Score 2:
- **Score 1**: The answer is **WRONG** (contradicts or uses external knowledge inappropriately)
- **Score 2**: The answer is **UNGROUNDED** (not based on retrieved content, insufficient, or made up, but not necessarily contradicting)

### The distinction between Score 4 and Score 5:
- **Score 4**: Answer is very good but missing some minor details or specificity
- **Score 5**: Answer is essentially equivalent to the ideal answer in all important ways

---

## Usage Notes

- This rubric is designed for evaluating RAG (Retrieval-Augmented Generation) chatbot systems
- Evaluators should have access to: the question, ideal answer, generated answer, and retrieved content
- Scores should be assigned independently by each evaluator before comparing
- Notes explaining the score are highly valuable for improving the system
- Focus on whether the generated answer would be helpful and accurate for the end user

---

## Document History
- Created: 2025-12-12
- Based on: Analysis of pathway-chatbot-answer-evaluation.csv
- Purpose: Standardize evaluation of BYU-Pathway chatbot responses
