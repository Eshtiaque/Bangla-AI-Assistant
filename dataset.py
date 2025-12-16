from langchain_core.documents import Document

def get_data():
    return [
        # Topic: Shiksha (শিক্ষা)
        Document(page_content="প্রশ্ন: এইচএসসি পরীক্ষা কবে হতে পারে? উত্তর: ২০২৫ সালের এপ্রিল মাসে।", metadata={"topic": "shiksha"}),
        Document(page_content="প্রশ্ন: ঢাকা বিশ্ববিদ্যালয় কোথায় অবস্থিত? উত্তর: এটি ঢাকার শাহবাগে অবস্থিত।", metadata={"topic": "shiksha"}),
        Document(page_content="প্রশ্ন: সিএসই (CSE) পড়তে কত বছর লাগে? উত্তর: সাধারণত ৪ বছরের বিএসসি কোর্স করতে হয়।", metadata={"topic": "shiksha"}),

        # Topic: Shastho (স্বাস্থ্য)
        Document(page_content="প্রশ্ন: জ্বরের ঘরোয়া চিকিৎসা কী? উত্তর: বেশি করে পানি পান করা এবং পূর্ণ বিশ্রাম নেওয়া।", metadata={"topic": "shastho"}),
        Document(page_content="প্রশ্ন: ডেঙ্গু জ্বরের লক্ষণ কী? উত্তর: তীব্র জ্বর, প্রচণ্ড মাথা ব্যথা, চোখের পেছনে ব্যথা ও শরীরে র‍্যাশ।", metadata={"topic": "shastho"}),
        Document(page_content="প্রশ্ন: গ্যাস্ট্রিকের সমস্যা কেন হয়? উত্তর: দীর্ঘক্ষণ না খেয়ে থাকলে, ভাজাপোড়া বা অতিরিক্ত তেলচর্বিযুক্ত খাবার খেলে।", metadata={"topic": "shastho"}),

        # Topic: Kheladhula (খেলাধুলা)
        Document(page_content="প্রশ্ন: বাংলাদেশ ক্রিকেট দলের বর্তমান ক্যাপ্টেন কে? উত্তর: নাজমুল হোসেন শান্ত।", metadata={"topic": "kheladhula"}),
        Document(page_content="প্রশ্ন: ফুটবল খেলার সময়কাল কতক্ষণ? উত্তর: মোট ৯০ মিনিট (৪৫ মিনিট করে দুই অর্ধ)।", metadata={"topic": "kheladhula"}),
        Document(page_content="প্রশ্ন: লিওনেল মেসি কোন দেশের খেলোয়াড়? উত্তর: তিনি আর্জেন্টিনার খেলোয়াড়।", metadata={"topic": "kheladhula"}),

        # Topic: Projukti (প্রযুক্তি)
        Document(page_content="প্রশ্ন: এআই (AI) বলতে কী বোঝায়? উত্তর: আর্টিফিশিয়াল ইন্টেলিজেন্স বা কৃত্রিম বুদ্ধিমত্তা, যা মানুষের মতো চিন্তা করতে পারে।", metadata={"topic": "projukti"}),
        Document(page_content="প্রশ্ন: র‍্যাম (RAM) এর কাজ কী? উত্তর: কম্পিউটারের চলমান ডাটা সাময়িকভাবে জমা রাখা বা স্টোর করা।", metadata={"topic": "projukti"}),
        Document(page_content="প্রশ্ন: পাইথন (Python) কী? উত্তর: পাইথন একটি অত্যন্ত জনপ্রিয় এবং সহজ প্রোগ্রামিং ল্যাঙ্গুয়েজ।", metadata={"topic": "projukti"}),

        # Topic: Vromon (ভ্রমণ)
        Document(page_content="প্রশ্ন: কক্সবাজার কেন বিখ্যাত? উত্তর: কারণ এখানে বিশ্বের দীর্ঘতম অখণ্ড প্রাকৃতিক সমুদ্র সৈকত রয়েছে।", metadata={"topic": "vromon"}),
        Document(page_content="প্রশ্ন: সাজেক ভ্যালি কোথায় অবস্থিত? উত্তর: এটি রাঙ্গামাটি জেলায় অবস্থিত, যাকে মেঘের রাজ্য বলা হয়।", metadata={"topic": "vromon"}),
        Document(page_content="প্রশ্ন: ভিসা ছাড়া বাংলাদেশিরা কোন দেশে যেতে পারে? উত্তর: ভুটান এবং কিছু দ্বীপ রাষ্ট্রে অন-অ্যারাইভাল ভিসায় যাওয়া যায়।", metadata={"topic": "vromon"}),
    ]