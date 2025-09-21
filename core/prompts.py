# Few shot prompts for different tasks

GSM8K_prompt = [
    {
        "role": "system",
        "content": "You are a prompt rewriter whose main goal is to rewrite the prompt given by the user in the most optimal way without losing any information in them."
    },
    {
        "role": "user",
        "content": (
            "I have a set of questions and/or statements, please REWRITE all the questions/statements so that they are in the most optimal order that is the easiest to understand. DO NOT ANSWER ANY OF THE QUESTIONS JUST REWRITE. Here are the instructions:\n"
            "how much will John pay in sales tax?\n"
            "the purchase amount is $120\n"
            "the sales tax rate is 8.25%."
        )
    },
    {
        "role": "assistant",
        "content": "What is the total sales tax John will pay on a $120 purchase at an 8.25% rate?"
    },
]

Code_prompt = [
    {
        "role": "system",
        "content": "You are a prompt rewriter whose main goal is to rewrite the prompt given by the user in the most optimal way without losing any information in them."
    },
    {
        "role": "user",
        "content": (
            "I have a set of questions and/or statements, please REWRITE all the questions/statements so that they are in the most optimal order that is the easiest to understand. DO NOT ANSWER ANY OF THE QUESTIONS JUST REWRITE. Here are the instructions:\n"
            "create a python function that takes two arguments"
            "check if both arguments are strings"
            "if not, raise TypeError"
            "return concatenated result"
        )
    },
    {
        "role": "assistant",
        "content": "Write a function that takes two arguments, checks if both are strings, raises a TypeError if not, and returns their concatenated result."
    },
]

DB_prompt = [
    {
        "role": "system",
        "content": "You are a prompt rewriter whose main goal is to rewrite the prompt given by the user in the most optimal way without losing any information in them."
    },
    {
        "role": "user",
        "content": (
            "I have a set of questions and/or statements, please REWRITE all the questions/statements so that they are in the most optimal order that is the easiest to understand. DO NOT ANSWER ANY OF THE QUESTIONS JUST REWRITE. Here are the instructions:\n"
            "Write SQL query to find active users"
            "active users are those who made at least 5 purchases"
            "each purchase is in the purchases table"
            "users are in the users table"
            "return user names only"
        )
    },
    {
        "role": "assistant",
        "content": "Write an SQL query to find the names of users who have made at least 5 purchases, using the users and purchases tables."
    },
]

D2T_prompt = [
    {
        "role": "system",
        "content": "You are a prompt rewriter whose main goal is to rewrite the prompt given by the user in the most optimal way without losing any information in them."
    },
    {
        "role": "user",
        "content": (
            "I have a set of questions and/or statements, please REWRITE all the questions/statements so that they are in the most optimal order that is the easiest to understand. DO NOT ANSWER ANY OF THE QUESTIONS JUST REWRITE. Here are the instructions:\n"
"""User Instruction 1: I am providing you a table. You must produce a short one-sentence description of the table. The description should be at most 30 words. In followup turns, I will provide further information about the table. You must update your description to take information from all the turns into account. At each turn, only respond with a single sentence describing the table. Do not respond with any other text.

Table:
<table>
<tr> <th colspan=1 rowspan=1 > Pos </th><th colspan=1 rowspan=1 > No. </th><th colspan=1 rowspan=1 > Driver </th><th colspan=1 rowspan=1 > Team </th><th colspan=1 rowspan=1 > Manufacturer </th><th colspan=1 rowspan=1 > Time </th><th colspan=1 rowspan=1 > Speed </th></tr>
<tr> <th colspan=1 rowspan=1 > 1 </th><td colspan=1 rowspan=1 > 41 </td><td colspan=1 rowspan=1 > Kurt Busch </td><td colspan=1 rowspan=1 > Stewart-Haas Racing </td><td colspan=1 rowspan=1 > Chevrolet </td><td colspan=1 rowspan=1 > 25.928 </td><td colspan=1 rowspan=1 > 138.846 </td></tr>
<tr> <th colspan=1 rowspan=1 > 2 </th><td colspan=1 rowspan=1 > 11 </td><td colspan=1 rowspan=1 > Denny Hamlin </td><td colspan=1 rowspan=1 > Joe Gibbs Racing </td><td colspan=1 rowspan=1 > Toyota </td><td colspan=1 rowspan=1 > 25.999 </td><td colspan=1 rowspan=1 > 138.467 </td></tr>
</table>

User Instruction 2: Here's 2 examples of descriptions of other tables. You can use it as references to understand how to describe the table:
Hagino's fellow countryman Daiya Seto captured the bronze in 4:09.71, to give Japan two swimmers on the same Olympic podium.
In 2017, Stamenković appeared in the Kazakhstan Premier League for Irtysh Pavlodar.
User Instruction 3: 'I am now giving you a version of the table with some cells highlights (by adding a highlighted class to each cell). You can now update your description of the table, which must now focus on the highlighted cells in the table.

<table>
<tr> <th colspan=1 rowspan=1 > Pos </th><th colspan=1 rowspan=1 > No. </th><th colspan=1 rowspan=1 > Driver </th><th colspan=1 rowspan=1 > Team </th><th colspan=1 rowspan=1 > Manufacturer </th><th colspan=1 rowspan=1 > Time </th><th colspan=1 rowspan=1 > Speed </th></tr>
<tr> <th class="highlighted" colspan=1 rowspan=1 > 1 </th><td colspan=1 rowspan=1 > 41 </td><td class="highlighted" colspan=1 rowspan=1 > Kurt Busch </td><td colspan=1 rowspan=1 > Stewart-Haas Racing </td><td colspan=1 rowspan=1 > Chevrolet </td><td colspan=1 rowspan=1 > 25.928 </td><td colspan=1 rowspan=1 > 138.846 </td></tr>
<tr> <th class="highlighted" colspan=1 rowspan=1 > 2 </th><td colspan=1 rowspan=1 > 11 </td><td class="highlighted" colspan=1 rowspan=1 > Denny Hamlin </td><td colspan=1 rowspan=1 > Joe Gibbs Racing </td><td colspan=1 rowspan=1 > Toyota </td><td colspan=1 rowspan=1 > 25.999 </td><td colspan=1 rowspan=1 > 138.467 </td></tr>
</table>

User Instruction 4: Oh also, Here's additional metadata about the table. The table_webpage_url is http://en.wikipedia.org/wiki/2016_Good_Sam_500.
User Instruction 5: FYI, Here's additional metadata about the table. The table_page_title is 2016 Good Sam 500."""
        )
    },
    {
        "role": "assistant",
        "content": """Provide a short one-sentence description of the following table. Only focus on the highlighted cells. In future turns, update the description with any new provided information. Output only one sentence per turn, no other text.

Table:
<table>
<tr> <th colspan=1 rowspan=1 > Pos </th><th colspan=1 rowspan=1 > No. </th><th colspan=1 rowspan=1 > Driver </th><th colspan=1 rowspan=1 > Team </th><th colspan=1 rowspan=1 > Manufacturer </th><th colspan=1 rowspan=1 > Time </th><th colspan=1 rowspan=1 > Speed </th></tr>
<tr> <th class="highlighted" colspan=1 rowspan=1 > 1 </th><td colspan=1 rowspan=1 > 41 </td><td class="highlighted" colspan=1 rowspan=1 > Kurt Busch </td><td colspan=1 rowspan=1 > Stewart-Haas Racing </td><td colspan=1 rowspan=1 > Chevrolet </td><td colspan=1 rowspan=1 > 25.928 </td><td colspan=1 rowspan=1 > 138.846 </td></tr>
<tr> <th class="highlighted" colspan=1 rowspan=1 > 2 </th><td colspan=1 rowspan=1 > 11 </td><td class="highlighted" colspan=1 rowspan=1 > Denny Hamlin </td><td colspan=1 rowspan=1 > Joe Gibbs Racing </td><td colspan=1 rowspan=1 > Toyota </td><td colspan=1 rowspan=1 > 25.999 </td><td colspan=1 rowspan=1 > 138.467 </td></tr>
</table>

Here is additional metadata about the table:
The table_webpage_url is http://en.wikipedia.org/wiki/2016_Good_Sam_500.
The table_page_title is 2016 Good Sam 500.

Here are 4 example descriptions for reference:
Hagino's fellow countryman Daiya Seto captured the bronze in 4:09.71, to give Japan two swimmers on the same Olympic podium.
In 2017, Stamenković appeared in the Kazakhstan Premier League for Irtysh Pavlodar."""
    },
]