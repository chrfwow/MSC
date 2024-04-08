from counterfactuals2.classifier.CodeReviewerClassifier import CodeReviewerClassifier

should_be_one = ["""
<s><start><add><?php<add>/*<add>* Copyright ▒ Bold Brand Commerce Sp. z o.o. All rights reserved.<add>* See LICENSE.txt for license details.<add>*/<add>declare(strict_types=1);<add>namespace Ergonode\SharedKernel\Domain\ValueObject;<add>use Ergonode\SharedKernel\Domain\AbstractCode;<add>class Code extends AbstractCode<add>{<add>}<end></s>
""".strip(),
                 """
<s><start><add>require "rails_helper"<add>feature "User reactivates subscription" do<add>scenario "after canceling, but before the grace period ends" do<add>sign_in_as_user_with_subscription<add>@current_user.subscription.update!(scheduled_for_deactivation_on: 1.month.from_now)<add>visit my_account_path<add>click_on I18n.t("subscriptions.reactivate")<add>expect(page).to have_content(I18n.t("subscriptions.flashes.reactivate.success"))<add>end<add>end<end></s>                 
                 """.strip(),
                 """
<s><start><add>package org.phoenicis.engines;<add>/**<add>* interface which must be implemented by all Verbs in Javascript<add>*/<add>public interface Verb {<add>/**<add>* installs the Verb in the given container<add>* @param container directory name (not the complete path!) of the container where the Verb shall be installed<add>* @param version version of the Verb<add>*/<add>void install(String container, String version);<add>}<end></s>                 
                 """.strip(),
                 """
<s>* @param programName the name of the program* @param programEditor the editor of the program* @param applicationHomepage homepage of the application* @param scriptorName the scriptor name*/<start><keep>@Override<keep>public Void presentation(String programName, String programEditor, String applicationHomepage,<keep>String scriptorName) {<del>final String htmlToShow = "<body>" + tr("This wizard will help you install {0} on your computer", programName)<del>+ ".<br><br>" + tr("This program was created by: {0}", programEditor) + "<br><br>"<add>final String htmlToShow = "<body>" + tr("Installation Wizard for {0}", programName)<add>+ ".<br><br>" + tr("Installation Script by {0}", programEditor) + "<br><br>"<keep>+ tr("For more information about this program, visit:")<keep>+ String.format("<br><a href=\"%1$s\">%1$s</a><br><br>", applicationHomepage)<del>+ tr("This installation program is provided by: {0}", scriptorName) + "<br><br>" + "<br><br>"<add>+ tr("Installation Script by {0}", scriptorName) + "<br><br>" + "<br><br>"<keep>+ tr("{0} will be installed in: {1}", programName, applicationUserRoot) + "<br><br>"<keep>+ tr("{0} is not responsible for anything that might happen as a result of using these scripts.",<keep>applicationName)<end>+ "<br><br>" + tr("Click \"Next\" to start.") + "</body>";return messageSender.runAndWait(message -> setupUi.showHtmlPresentationStep(message, htmlToShow));}/**</s>                 
                 """.strip()]

classifier = CodeReviewerClassifier()

for string in should_be_one:
    output = classifier.classify(string, True)
    current_classification, score = output[0] if isinstance(output, list) else output
    print("with tokens: should be 1", "current_classification", current_classification[0], "score", score)

    string = string.replace("<start>", "").replace("<s>", "").replace("<keep>", "\n").replace("<del>", "\n").replace(
        "<end>", "").replace("<add>", "\n").replace("</s>", "")
    output = classifier.classify(string, True)
    current_classification, score = output[0] if isinstance(output, list) else output
    print("without tokens: should be 1", "current_classification", current_classification[0], "score", score)
