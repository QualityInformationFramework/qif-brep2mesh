def write_obj(construct, file_name, metadata = False): 

    with open(file_name, "w") as file:
        if metadata:
            for key, value in metadata.items():
                file.write(f"# {key} {value}\n")

        for point in construct['points']:
            file.write('v ' + ' '.join(map(str, point)) + "\n")

        for face in construct['faces']:
            file.write('g ' + str(face['id']) + "\n")
            for tri in face['triangles'] + 1:
                file.write('f ' + ' '.join(map(str, tri)) + "\n")
