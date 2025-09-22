import json

# Base system prompts are taken directly from Laban et al.

class Dataset():
    def __init__(self, dataset_name, dataset_path):
        """
        Initialize instances of Datasets. 
        Each dataset has its own base system prompt, some have extra final shard instructions.
        """

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.data = []
        self.final_shard_instruct = ""
        # For more realistic humanlike transition between shards:
        self.connectors = ["oh also, ", "I just remembered, ", "sorry i forgot to say, ", "", "oh, and ", "FYI, "]

    def get_base_system(self, i):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

class GSM8K(Dataset):
    def __init__(self, dataset_path):
        super().__init__("GSM8K", dataset_path)
        self.data = self.load_data()
        self.final_shard_instruct = ""
    
    def get_base_system(self, i):
        return {
        "role": "system",
        "content": (
            "You are a helpful math assistant."
            )
        }

    def load_data(self):
        data = []
        with open(self.dataset_path, 'r') as f:
            temp_data = json.load(f)
        
        for task in temp_data:
            if 'math' in task['task']:
                data.append(task)

        return data
    
class Database(Dataset):
    def __init__(self, dataset_path):
        super().__init__("Database", dataset_path)
        self.data = self.load_data()
        self.final_shard_instruct = " Include your complete new Query in your response."

    def get_base_system(self, i):
        return {
            "role": "system",
            "content": (
                f"""\nYou are helping a user write SQL queries to a database. If something is not clear, you can ask the user to clarify what they need. The schema for the database being accessed is the following:\n{self.data[i]['schema_sql']}"""
            )
        }
    
    def load_data(self):
        data = []
        with open(self.dataset_path, 'r') as f:
            temp_data = json.load(f)
        
        for task in temp_data:
            if 'database' in task['task']:
                data.append(task)

        return data

class Code(Dataset):
    def __init__(self, dataset_path):
        super().__init__("Code", dataset_path)
        self.data = self.load_data()
        self.final_shard_instruct = " Please include your entire Python Function in your response."

    def get_base_system(self, i):
        return {
        "role": "system",
        "content": (
                f"""You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.

                Format:
                - [Standalone] Make sure that your answer consists of only one Python function at the top level. Do not wrap with a class or split into multiple functions."""
                )
        }

    def load_data(self):
        data = []
        with open(self.dataset_path, 'r') as f:
            temp_data = json.load(f)
        
        for task in temp_data:
            if 'code' in task['task']:
                data.append(task)

        return data
    
class Actions(Dataset):
    def __init__(self, dataset_path):
        super().__init__("Actions", dataset_path)
        self.data = self.load_data()
        self.final_shard_instruct = " Please include all the functions neccessary to complete my task in your response."


    def get_base_system(self, i):
        return {
        "role": "system",
        "content": (
            f"""You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
                If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
                You should only return the function calls in your response.

                If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
                You SHOULD NOT include any other text in the response.

                At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.

                Here is a list of functions in JSON format that you can invoke.

                {self.data[i]['function']}"""
            )
        }

    def load_data(self):
        data = []
        with open(self.dataset_path, 'r') as f:
            temp_data = json.load(f)
        
        for task in temp_data:
            if 'actions' in task['task']:
                data.append(task)

        return data
    
class DataToText(Dataset):
    def __init__(self, dataset_path):
        super().__init__("DataToText", dataset_path)
        self.data = self.load_data()
        self.final_shard_instruct = ""

    def get_base_system(self, i):
        return {
        "role": "system",
        "content": (
            f"You are an analyst with an eye for detail that accomplishes tasks carefully and thoroughly."
        )
        }


    def load_data(self):
        data = []
        with open(self.dataset_path, 'r') as f:
            temp_data = json.load(f)
        
        for task in temp_data:
            if 'data2text' in task['task']:
                data.append(task)

        return data