instruction = """Your task is to analyse and critique the steps in the solution of a math problem step-by-step.
Please follow the instructions and format below to respond:

1. Verify the computational expression explicitly written in the step using the Wolfram Alpha self-questionnaire with the format: <verify>\n The expression I needed to verify is: [expression], the result is: [output] \n</verify>.
2. Judge the logical correctness of the step by yourself, or the calculation correctness of the step by using the results of the wolfram alpha as a reference if the step has a definite result, using the format of <judge>\n [reason] \n</judge>.
    - The correctness of the Wolfram Alpha does not affect the label in the output.
    - WA only serves as a reference for final result verification. If the step does not produce a definite result, it can still be considered correct as long as the intermediate reasoning is valid, even if incomplete
3. Provide the judgement conclusion and the label as "correct" or "incorrect" in '<output>\\boxed{}</output>'.
    - If the result of the judge confirmed that wolfram alpha and the input are match, or the judge ensure the input is correct without a relevant wolfram alpha result, set the label to "correct".
    - Else set the label to "incorrect".

Here are the math problem and steps:"""