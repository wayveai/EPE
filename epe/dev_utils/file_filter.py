# %%
import os
# %%
def remove_lines(file_path, excluded_keywords):
    temp_output = '/tmp/file_filter.txt'
    with open(file_path, 'r') as file_in:
        with open(temp_output, "w") as file_out:
            file_out.writelines(filter(lambda line: not any([x in line for x in excluded_keywords]), file_in))

    os.replace(temp_output, file_path)

def keep_lines(file_path, out_file_path, keywords):
    with open(file_path, 'r') as file_in:
        with open(out_file_path, "w") as file_out:
            file_out.writelines(filter(lambda line: any([x in line for x in keywords]), file_in))

# %%

# dark_worlds = [
# '2022-10-05--18-52-55--somerstown-aft-loop-anti-clockwise-v1--c92624ba44dfa9c9--44c44ade',
# '2022-10-05--21-09-30--somerstown-aft-loop-anti-clockwise-v1--5b4e723801f6a739--58281ec4',
# '2022-10-06--05-10-54--somerstown-aft-loop-clockwise-v1--f56e08f1fa0404db--25c23d78',
# '2022-10-06--04-01-47--somerstown-aft-loop-clockwise-v1--ea2e41b460f96ae7--9dedfe93',
# ]

# files = ['/home/kacper/data/EPE/somers_town/sim_files.csv']
# for file in files:
#     remove_lines(file, dark_worlds)


# %%
file = '/home/kacper/data/EPE/somers_town/sim_files.csv'
runs = ['2022-10-05--13-11-03--somerstown-aft-loop-anti-clockwise-v1--f0e9b72c9fb9bb9f--5aae8a99']
keep_lines(file, '/home/kacper/data/EPE/somers_town/video_test.csv', runs)

# %%
