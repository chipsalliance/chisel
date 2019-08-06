import subprocess, os

output = subprocess.check_output(["sbt", "clean", "compile"]).decode('utf-8')
    
def chisel_rst_text(filename):
    file = open(filename)
    contents = file.read()
    lines = contents.split('\n')
    comments = []
    for i in range(len(lines)): 
        line = lines[i]
        if len(line) >= 3 and line.strip()[0:3] == '/**':
            j = i
            line = line.strip()[3:].strip()
            line = line.replace('[[', ':chisel:reref:`')
            line = line.replace(']]', '` ')
            comment = ""
            end_of_last_sig = -1
            if '*/' in line:
                index = line.index('*/')
                comment += '\n\t' + line[0:index] + '\n'
            else:
                comment += '\n\t' + line
                is_code_block = False
                is_param_block = False
                is_return_block = False
                is_note_block = False
                is_io_block = False
                is_io_attr_block = False
                curr_io_name = ""
                io_dict = {}
                while not '*/' in lines[j]:
                    comment_line = lines[j].strip()
                    comment_line = comment_line.replace('[[', ':chisel:reref:`')
                    comment_line = comment_line.replace(']]', '` ')
                    if comment_line != '' and comment_line[0] == '*':
                        if not is_code_block:
                            comment_line = comment_line[1:].strip()
                        if '{{{' in comment_line:
                            is_code_block = True
                            is_param_block = False
                            is_return_block = False
                            is_note_block = False
                            is_io_block = False
                            is_io_attr_block = False
                            comment_line = "\n\t.. code-block:: scala \n"
                        if is_code_block:
                            if comment_line != '' and comment_line[0] == '*':
                                comment_line = '\t' + comment_line[1:]
                        if '}}}' in comment_line:
                            is_code_block = False 
                            comment_line = ""
                        if '@param ' in comment_line or '@tparam ' in comment_line:
                            is_param_block = True
                            is_return_block = False
                            is_note_block = False
                            is_io_block = False
                            is_io_attr_block = False
                            tag = '@tparam ' if '@tparam ' in comment_line else '@param '
                            index = comment_line.index(tag) + 7
                            rest_of_line = comment_line[index:].strip()
                            space = rest_of_line.index(' ')
                            rest_of_line = rest_of_line[0:space] + ':' + rest_of_line[space:]
                            middle_str = '\t:type-param ' if '@tparam ' in comment_line else '\t:param '
                            comment_line = "\n" + comment_line[0:index-7] + middle_str + rest_of_line
                        elif '@io ' in comment_line:
                            is_param_block = False
                            is_return_block = False
                            is_note_block = False
                            is_io_block = True
                            is_io_attr_block = False
                            comment_line = comment_line.replace('@io', ':IO:').strip()
                            io_name = comment_line.split()[1]
                            full_name = 'val ' + io_name + ': '
                            val_line = get_val_line(io_name[0])
                            index = val_line.index(full_name) + len(full_name)
                            val_class = val_line[index:].strip().split(' ')[0]
                            if '.' in val_class:
                                val_class = val_class[val_class.rindex('.')+1:]
                            class_lines = get_all_class_lines(val_class)
                            type_dict = {}
                            if class_lines != None:
                                for line in class_lines:
                                    if ' val ' in line:
                                        ind = line.index('val')
                                        line = line[ind + 3:].strip()
                                        tokens = line.split()
                                        type_dict[tokens[0][0:len(tokens[0])-1]] = tokens[1]
                            io_dict = type_dict
                        elif '@return ' in comment_line:
                            is_return_block = True
                            is_param_block = False
                            is_note_block = False
                            is_io_block = False
                            is_io_attr_block = False
                            comment_line = comment_line.replace('@return', ':return:').strip()
                        elif '@note' in comment_line:
                            is_return_block = False
                            is_param_block = False
                            is_note_block = True
                            is_io_block = False
                            is_io_attr_block = False
                            comment_line = '\n\t' + comment_line.replace('@note', ':note:').strip()
                        elif '@ioattr' in comment_line:
                            is_return_block = False
                            is_param_block = False
                            is_note_block = False
                            is_io_block = False
                            is_io_attr_block = True
                            tokens = comment_line.split()
                            name = tokens[1]
                            comment_line = comment_line.replace('@ioattr', ':IO\xa0attr:')
                            if tokens[2] != ':':
                                comment_line = comment_line.replace(name, name + ' :chisel:reref:`' + find_io_attribute_type(name) + '` -')
                            else:
                                comment_line = comment_line.replace(tokens[2] + ' ' + tokens[3], ':chisel:reref:`' + tokens[3] + '` -')
                        elif is_return_block or is_param_block or is_note_block or is_io_block:
                            comment_line = '\t' + comment_line
                        comment += '\t' + comment_line + '\n'
                    j += 1
                index = lines[j].index("*/")
                last_line = lines[j][0:index]
                if '@return ' in last_line:
                    is_return_block = True
                    is_param_block = False
                    last_line.replace('@return', ':return:')
                elif '@param ' in comment_line:
                    is_param_block = True
                    is_return_block = False
                    index = last_line.index('@param ') + 7
                    rest_of_line = last_line[index:]
                    space = rest_of_line.index(' ')
                    rest_of_line = rest_of_line[0:space] + ':' + rest_of_line[space:]
                    last_line = "\n" + last_line[0:index-7] + '\t:param ' + rest_of_line
                elif is_return_block or is_param_block:
                    last_line = "\t" + last_line
                comment += last_line
            while lines[j+1].strip() == "" or lines[j+1].strip()[0:2] == "//" or lines[j+1].strip()[0] == '@':
                j += 1
            k = j + 1
            signature = ""
            while k < len(lines) and not ((contains_sig(lines[k]) or '{' in lines[k]) and k > j + 1) and lines[k] != "":
                line_k = lines[k].strip()
                if line_k[0:2] != '//' and '//' in line_k:
                    line_k = line_k[0:line_k.index('//')]
                signature += line_k.strip()
                if signature[-1] != '(':
                    signature += ' '
                k += 1
            if k < len(lines):
                signature += lines[k].strip()
            signature.strip()
            if signature[-1] == '=':
                signature = signature[0:len(signature)-1]
            '''
            signature = lines[j+1].strip()
            if "{" in signature:
                ind = signature.index("{")
                signature = signature[0:ind].strip()
            elif "= " in signature:
                ind = signature.index("= ")
                signature = signature[0:ind].strip()
            elif signature.strip()[len(signature)-1] == '=':
                signature = signature.strip()[0:len(signature)-1].strip()
            '''
            if '{' in signature:
                signature = signature[0:signature.index('{')].strip()
            if '}' in signature:
                signature = signature[0:signature.index('}')].strip()
            if ' = ' in signature:
                index = signature.index(' = ')
                substr = signature[0:index]
                if substr.count('(') == substr.count(')'):
                    signature = signature[0:index].strip()
            if '//' in signature:
                index = signature.index('//')
                signature = signature[0:index].strip()
            if '/*' in signature:
                index = signature.index('/*')
                signature = signature[0:index].strip()
            if 'def' in signature.split() or 'class' in signature.split() or 'object' in signature.split() or 'trait' in signature.split():
                comments += [(comment, signature.strip())]
        elif ('class' in line.split(" ") or 'object' in line.split(" ") or 'trait' in line.split(" ")) and line.strip()[0:2] != '//' and line.strip()[0:2] != '/*' and line[0] != '\t' and line.strip()[0] != '*' and line[0] != ' ':
            signature = line
            k = i
            signature = ""
            while k < len(lines) and not ((contains_sig(lines[k]) or '{' in lines[k] or '//' in lines[k] or '/*' in lines[k]) and k > i + 1):
                line_k = lines[k].strip()
                if line_k[0:2] != '//' and '//' in line_k:
                    line_k = line_k[0:line_k.index('//')]
                signature += line_k.strip()
                if signature[-1] != '(':
                    signature += ' '
                k += 1
            if k < len(lines):
                signature += lines[k].strip()
            if '{' in signature:
                signature = signature[0:signature.index('{')].strip()
            if '}' in signature:
                signature = signature[0:signature.index('}')].strip()
            if ' = ' in signature:
                index = signature.index(' = ')
                signature = signature[0:index].strip()
            if '//' in signature:
                index = signature.index('//')
                signature = signature[0:index].strip()
            if '/*' in signature:
                index = signature.index('/*')
                signature = signature[0:index].strip()
            lst = [c[1] for c in comments]
            if not signature in lst:
                comments += [("", signature)]
    rst_text = ""
    '''
    if '/' in filename:
        ind = filename.rindex('/')
        filename_no_dir = filename[ind+1:]
        rst_text += '-' * len(filename_no_dir) + '\n'
        rst_text += filename_no_dir + '\n'
        rst_text += '-' * len(filename_no_dir) + '\n'
    else:
        rst_text += '-' * len(filename) + '\n'
        rst_text += filename + '\n'
        rst_text += '-' * len(filename) + '\n'
    '''
    for comm, sig in comments:
        if 'class ' in sig or 'object ' in sig or 'trait ' in sig:
            rst_text += ".. chisel:attr:: " + sig + "\n" + comm + "\n\n"
        else:
            rst_text += "\t.. chisel:attr:: " + sig + '\n\n'
            lines = comm.split('\n')
            for line in lines:
                rst_text += '\t' + line + '\n'
            rst_text += '\n\n'
    return rst_text

def contains_sig(line):
    words = line.strip().split(' ')
    return 'def' in words or 'trait' in words or 'class' in words or 'object' in words

# Returns lines that define vals.
def get_val_lines():
    lines = output.split('\n')
    val_lines = []
    for line in lines:
        if ' val ' in line:
            val_lines.append(line)
    return val_lines

# Gets the line where an IO name is defined.  
def get_val_line(io_name):
    val_lines = get_val_lines()
    contains_io_name = [line for line in val_lines if "val " + io_name in line][0]
    return contains_io_name

def get_all_class_lines(class_name):
    lines = output.split('\n')
    important_lines = []
    for i in range(0, len(lines)):
        if 'class ' + class_name in lines[i]:
            j = 0
            while '{' not in lines[i+j]:
                important_lines.append(lines[i+j])
                j += 1
            important_lines.append(lines[i+j])
            leftrightdiff = lines[i+j].count('{') - lines[i+j].count('}')
            while leftrightdiff > 0:
                j += 1
                leftrightdiff += lines[i+j].count('{') - lines[i+j].count('}')
                if 'val ' in lines[i+j]:
                    important_lines.append(lines[i+j])
            return important_lines

def find_io_attribute_type(io_attr_full_name):
    items_list = io_attr_full_name.split('.')
    type_dict = {}
    for i in range(0, len(items_list)):
        if i == 0:
            val_line = get_val_line(items_list[0])
            full_name = 'val ' + items_list[0] + ': '
            index = val_line.index(full_name) + len(full_name)
            val_class = val_line[index:].strip().split(' ')[0]
            if '.' in val_class:
                val_class = val_class[val_class.rindex('.')+1:]
            class_lines = get_all_class_lines(val_class)
            if class_lines != None:
                for line in class_lines:
                    if ' val ' in line:
                        ind = line.index('val')
                        line = line[ind + 3:].strip()
                        tokens = line.split()
                        type_dict[tokens[0][0:len(tokens[0])-1]] = tokens[1]
        else:
            typ = type_dict.get(items_list[i])
            if i == len(items_list) - 1:
                return typ
            else:
                type_dict = {}
                type_dots = typ.split('.')
                typ = type_dots[len(type_dots)-1]
                class_lines = get_all_class_lines(typ)
                if class_lines != None:
                    for line in class_lines:
                        if ' val ' in line:
                            ind = line.index('val')
                            line = line[ind + 3:].strip()
                            tokens = line.split()
                            type_dict[tokens[0][0:len(tokens[0])-1]] = tokens[1]

def generate_all_rst(directory):
    dirs = list(os.walk(directory))
    for directory, folders, files in dirs:
        toctree = '-' * len(directory) + '\n'
        toctree += directory + '\n'
        toctree += '-' * len(directory) + '\n\n'
        toctree += ".. toctree::\n"
        for folder in folders:
            toctree += '\t' + folder + '/' + folder + '.rst\n'
        rsts = ''
        for file in files:
            if file[len(file)-6:len(file)] == '.scala':
                rsts += file + '\n' + '-' * len(file) + '\n'
                rsts += chisel_rst_text(directory + '/' + file)
        dir_folder = directory
        if ('/' in directory):
            dir_folder = directory[directory.rindex('/')+1:]
        if not os.path.exists('sphinx-chisel-workspace/sample/source/' + directory):
            os.mkdir('sphinx-chisel-workspace/sample/source/' + directory)
        rst_output = open('sphinx-chisel-workspace/sample/source/' + directory + '/' + dir_folder + ".rst", 'w')
        rst_output.write(toctree + '\n\n' + rsts)
        rst_output.close()

def generate_sphinx():
    os.chdir('sphinx-chisel-workspace/sample')
    subprocess.run(['make', 'html'])

generate_all_rst('src')
generate_all_rst('chiselFrontend')
generate_sphinx()
