from counterfactuals2.classifier.CodeReviewerClassifier import CodeReviewerClassifier

should_be_zero = ["""
<s>r.URL.Path = "/"}return fs.serveFile(w, r, r.URL.Path)}// serveFile writes the specified file to the HTTP response.// name is '/'-separated, not filepath.Separator.func (fs FileServer) serveFile(w http.ResponseWriter, r *http.Request, name string) (int, error) {location := name// Prevent absolute path access on Windows.// TODO remove when stdlib http.Dir fixes this.if runtime.GOOS == "windows" {if filepath.IsAbs(name[1:]) {return http.StatusNotFound, nil}}<start><keep>f, err := fs.Root.Open(name)<keep>if err != nil {<add>// TODO: remove when http.Dir handles this<add>// Go issue #18984<add>err = mapFSRootOpenErr(err)<keep>if os.IsNotExist(err) {<keep>return http.StatusNotFound, nil<keep>} else if os.IsPermission(err) {<end>return http.StatusForbidden, err}// Likely the server is under load and ran out of file descriptorsbackoff := int(3 + rand.Int31()%3) // 3▒5 seconds to prevent a stampedew.Header().Set("Retry-After", strconv.Itoa(backoff))return http.StatusServiceUnavailable, err}defer f.Close()d, err := f.Stat()if err != nil {if os.IsNotExist(err) {return http.StatusNotFound, nil} else if os.IsPermission(err) {return http.StatusForbidden, err</s>
""".strip(),
                  """
 <s><start><keep>}<del>/**<del>* Write a list header.<del>*/<add>/// <summary><add>/// Write a list header.<add>/// </summary><keep>public override void WriteListBegin(TList list)<keep>{<keep>WriteCollectionBegin(list.ElementType, list.Count);<keep>}<del>/**<del>* Write a set header.<del>*/<add>/// <summary><add>/// Write a set header.<add>/// </summary><keep>public override void WriteSetBegin(TSet set)<keep>{<keep>WriteCollectionBegin(set.ElementType, set.Count);<keep>}<del>/**<del>* Write a boolean value. Potentially, this could be a boolean field, in<del>* which case the field header info isn't written yet. If so, decide what the<del>* right type header is for the value and then Write the field header.<del>* Otherwise, Write a single byte.<del>*/<add>/// <summary><add>/// Write a boolean value. Potentially, this could be a boolean field, in<add>/// which case the field header info isn't written yet. If so, decide what the<add>/// right type header is for the value and then Write the field header.<add>/// Otherwise, Write a single byte.<add>/// </summary><keep>public override void WriteBool(Boolean b)<keep>{<keep>if (booleanField_ != null)<end></s>
                  """.strip(),
                  """
 <s>* is at index {@code 0}, the next at index {@code 1},* and so on, as for array indexing.** <p>If the {@code char} value specified by the index is a* <a href="Character.html#unicode">surrogate</a>, the surrogate* value is returned.** @param index the index of the {@code char} value.* @return the {@code char} value at the specified index of this string.* The first {@code char} value is at index {@code 0}.* @throws IndexOutOfBoundsException if the {@code index}*                                   argument is negative or not less than the length of this*                                   string.<start><keep>*/<keep>@Override<keep>public char charAt(int index) {<del>return back.charAt(index);<add>return get(index);<keep>}<keep>/**<end>* Returns the length of this string.* The length is equal to the number of <a href="Character.html#unicode">Unicode* code units</a> in the string.** @return the length of the sequence of characters represented by this* object.*/@Overridepublic int length() {return back.length();}////</s>                 
                  """.strip(),
                  """
                  
                  """.strip()]

classifier = CodeReviewerClassifier()

for string in should_be_zero:
    print("#######################################################\n")
    output = classifier.classify(string, True)
    current_classification, score = output[0] if isinstance(output, list) else output
    print("with tokens: should be 0", "current_classification", current_classification[0], "score", score,
          "\nclassified:\n", string)

    string = string.replace("<start>", "").replace("<s>", "").replace("<keep>", "\n").replace("<del>", "\n").replace(
        "<end>", "").replace("<add>", "\n").replace("</s>", "")
    output = classifier.classify(string, True)
    current_classification, score = output[0] if isinstance(output, list) else output
    print("without tokens: should be 0", "current_classification", current_classification[0], "score", score,
          "\nclassified:\n", string)
